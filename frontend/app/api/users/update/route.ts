import { NextRequest, NextResponse } from "next/server";
import pool from "@/lib/db";

export async function PUT(request: NextRequest) {
  try {
    const { firebaseUid, displayName } = await request.json();

    if (!firebaseUid) {
      return NextResponse.json(
        { error: "Missing firebase UID" },
        { status: 400 }
      );
    }

    const result = await pool.query(
      `UPDATE users 
       SET display_name = $2, updated_at = CURRENT_TIMESTAMP
       WHERE firebase_uid = $1
       RETURNING id, firebase_uid, email, display_name, updated_at`,
      [firebaseUid, displayName]
    );

    if (result.rows.length === 0) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    return NextResponse.json({ user: result.rows[0] });
  } catch (error) {
    console.error("Error updating user:", error);
    return NextResponse.json(
      { error: "Failed to update user" },
      { status: 500 }
    );
  }
}
