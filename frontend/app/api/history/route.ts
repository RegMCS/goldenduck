import { NextRequest, NextResponse } from "next/server";
import pool from "@/lib/db";

export async function GET() {
  try {
    const result = await pool.query(
      `SELECT * FROM generation_history ORDER BY created_at DESC LIMIT 50`
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
    const { config, scenarioName } = body;

    if (!config) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    const result = await pool.query(
      `INSERT INTO generation_history (
        scenario_name, asset_class, trend_type, market_regime,
        volatility_level, generation_model, time_horizon, data_points,
        correlation_strength, mean_reversion, fat_tails, jump_diffusion, config_json
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
      RETURNING *`,
      [
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

    if (!id) {
      return NextResponse.json(
        { error: "Missing required parameters" },
        { status: 400 }
      );
    }

    const result = await pool.query(
      `DELETE FROM generation_history WHERE id = $1 RETURNING id`,
      [id]
    );

    if (result.rows.length === 0) {
      return NextResponse.json(
        { error: "Record not found" },
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
