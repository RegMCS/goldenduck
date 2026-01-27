import { NextRequest, NextResponse } from "next/server";
import pool from "@/lib/db";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const firebaseUid = searchParams.get("uid");

    if (!firebaseUid) {
      return NextResponse.json(
        { error: "Missing user ID" },
        { status: 400 }
      );
    }

    // Get user's generation history
    const result = await pool.query(
      `SELECT gh.* 
       FROM generation_history gh
       JOIN users u ON gh.user_id = u.id
       WHERE u.firebase_uid = $1
       ORDER BY gh.created_at DESC
       LIMIT 50`,
      [firebaseUid]
    );

    return NextResponse.json({ history: result.rows });
  } catch (error) {
    console.error("Error fetching history:", error);
    return NextResponse.json(
      { error: "Failed to fetch history" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { firebaseUid, config, scenarioName } = body;

    if (!firebaseUid || !config) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    // Get user ID from firebase UID
    const userResult = await pool.query(
      "SELECT id FROM users WHERE firebase_uid = $1",
      [firebaseUid]
    );

    if (userResult.rows.length === 0) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const userId = userResult.rows[0].id;

    // Insert generation history
    const result = await pool.query(
      `INSERT INTO generation_history (
        user_id, scenario_name, asset_class, trend_type, market_regime,
        volatility_level, generation_model, time_horizon, data_points,
        correlation_strength, mean_reversion, fat_tails, jump_diffusion, config_json
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
      RETURNING *`,
      [
        userId,
        scenarioName || `Scenario ${new Date().toISOString()}`,
        config.assetClass,
        config.trendType,
        config.marketRegime,
        config.volatilityLevel,
        config.generationModel,
        config.timeHorizon,
        config.dataPoints,
        config.correlationStrength,
        config.meanReversion,
        config.fatTails,
        config.jumpDiffusion,
        JSON.stringify(config),
      ]
    );

    return NextResponse.json({ record: result.rows[0] });
  } catch (error) {
    console.error("Error saving history:", error);
    return NextResponse.json(
      { error: "Failed to save history" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get("id");
    const firebaseUid = searchParams.get("uid");

    if (!id || !firebaseUid) {
      return NextResponse.json(
        { error: "Missing required parameters" },
        { status: 400 }
      );
    }

    // Verify ownership and delete
    const result = await pool.query(
      `DELETE FROM generation_history gh
       USING users u
       WHERE gh.user_id = u.id 
       AND u.firebase_uid = $1 
       AND gh.id = $2
       RETURNING gh.id`,
      [firebaseUid, id]
    );

    if (result.rows.length === 0) {
      return NextResponse.json(
        { error: "Record not found or unauthorized" },
        { status: 404 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting history:", error);
    return NextResponse.json(
      { error: "Failed to delete history" },
      { status: 500 }
    );
  }
}
