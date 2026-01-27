import { NextRequest, NextResponse } from "next/server";
import pool from "@/lib/db";

export async function POST(request: NextRequest) {
  try {
    const { firebaseUid, email, displayName } = await request.json();

    if (!firebaseUid || !email) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    // Upsert user
    const result = await pool.query(
      `INSERT INTO users (firebase_uid, email, display_name)
       VALUES ($1, $2, $3)
       ON CONFLICT (firebase_uid) 
       DO UPDATE SET 
         email = EXCLUDED.email,
         display_name = COALESCE(EXCLUDED.display_name, users.display_name),
         updated_at = CURRENT_TIMESTAMP
       RETURNING id, firebase_uid, email, display_name, created_at`,
      [firebaseUid, email, displayName]
    );

    return NextResponse.json({ user: result.rows[0] });
  } catch (error) {
    console.error("Error syncing user:", error);
    return NextResponse.json(
      { error: "Failed to sync user" },
      { status: 500 }
    );
  }
}
