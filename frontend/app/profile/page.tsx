"use client";

import React from "react"

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/auth-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  AlertCircle,
  Loader2,
  User,
  Mail,
  CheckCircle2,
  ArrowLeft,
  Clock,
  Trash2,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  BarChart3,
  RefreshCw,
} from "lucide-react";
import type { MarketConfig } from "@/lib/types";

interface HistoryRecord {
  id: number;
  scenario_name: string;
  asset_class: string;
  trend_type: string;
  market_regime: string;
  volatility_level: string;
  generation_model: string;
  time_horizon: number;
  data_points: number;
  created_at: string;
  config_json: MarketConfig;
}

const trendIcons: Record<string, React.ReactNode> = {
  bullish: <TrendingUp className="h-4 w-4 text-green-500" />,
  bearish: <TrendingDown className="h-4 w-4 text-red-500" />,
  sideways: <Minus className="h-4 w-4 text-yellow-500" />,
  volatile: <Zap className="h-4 w-4 text-orange-500" />,
};

const regimeColors: Record<string, string> = {
  normal: "bg-green-500/20 text-green-400 border-green-500/30",
  crisis: "bg-red-500/20 text-red-400 border-red-500/30",
  "high-volatility": "bg-orange-500/20 text-orange-400 border-orange-500/30",
  "low-volatility": "bg-blue-500/20 text-blue-400 border-blue-500/30",
  recovery: "bg-teal-500/20 text-teal-400 border-teal-500/30",
};

export default function ProfilePage() {
  const { user, loading: authLoading, updateDisplayName, signOut } = useAuth();
  const router = useRouter();
  const [displayName, setDisplayName] = useState("");
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  useEffect(() => {
    if (!authLoading && !user) {
      router.push("/login");
    }
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user?.displayName) {
      setDisplayName(user.displayName);
    }
  }, [user]);

  useEffect(() => {
    if (user?.uid) {
      fetchHistory();
    }
  }, [user?.uid]);

  const fetchHistory = async () => {
    if (!user?.uid) return;
    setLoadingHistory(true);
    try {
      const response = await fetch(`/api/history?uid=${user.uid}`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.history || []);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleSave = async () => {
    setError("");
    setSuccess("");
    setSaving(true);

    try {
      await updateDisplayName(displayName);
      setSuccess("Profile updated successfully");
      setEditing(false);
    } catch {
      setError("Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!user?.uid) return;
    setDeletingId(id);

    try {
      const response = await fetch(`/api/history?id=${id}&uid=${user.uid}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setHistory((prev) => prev.filter((item) => item.id !== id));
      }
    } catch (err) {
      console.error("Failed to delete:", err);
    } finally {
      setDeletingId(null);
    }
  };

  const handleSignOut = async () => {
    await signOut();
    router.push("/login");
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatLabel = (str: string) => {
    return str
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
              <Activity className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold text-foreground">SynthMarket</span>
          </Link>
          <div className="flex items-center gap-3">
            <Link href="/">
              <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Generator
              </Button>
            </Link>
            <Button
              variant="outline"
              size="sm"
              onClick={handleSignOut}
              className="border-border text-muted-foreground hover:text-foreground bg-transparent"
            >
              Sign out
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto max-w-6xl px-4 py-8">
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Profile Card */}
          <div className="lg:col-span-1">
            <Card className="border-border bg-card sticky top-24">
              <CardHeader>
                <CardTitle className="text-foreground">Profile</CardTitle>
                <CardDescription className="text-muted-foreground">
                  Manage your account settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {success && (
                  <Alert className="bg-primary/10 border-primary/50">
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                    <AlertDescription className="text-foreground">{success}</AlertDescription>
                  </Alert>
                )}

                {error && (
                  <Alert variant="destructive" className="bg-destructive/10 border-destructive/50">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <div className="flex flex-col items-center gap-4">
                  <div className="flex h-20 w-20 items-center justify-center rounded-full bg-secondary">
                    <User className="h-10 w-10 text-muted-foreground" />
                  </div>
                  {!editing && (
                    <div className="text-center">
                      <h3 className="text-lg font-semibold text-foreground">
                        {user.displayName || "User"}
                      </h3>
                      <p className="text-sm text-muted-foreground">{user.email}</p>
                    </div>
                  )}
                </div>

                {editing ? (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="displayName" className="text-foreground">
                        Display Name
                      </Label>
                      <div className="relative">
                        <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                        <Input
                          id="displayName"
                          value={displayName}
                          onChange={(e) => setDisplayName(e.target.value)}
                          className="pl-10 bg-secondary border-border text-foreground"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-foreground">Email</Label>
                      <div className="relative">
                        <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                        <Input
                          value={user.email || ""}
                          disabled
                          className="pl-10 bg-secondary/50 border-border text-muted-foreground"
                        />
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <Button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90"
                      >
                        {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save"}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setEditing(false);
                          setDisplayName(user.displayName || "");
                        }}
                        className="flex-1 border-border text-foreground"
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <Button
                      onClick={() => setEditing(true)}
                      variant="outline"
                      className="w-full border-border text-foreground hover:bg-secondary"
                    >
                      Edit Profile
                    </Button>
                    <Link href="/reset-password" className="block">
                      <Button
                        variant="ghost"
                        className="w-full text-muted-foreground hover:text-foreground"
                      >
                        Reset Password
                      </Button>
                    </Link>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* History Section */}
          <div className="lg:col-span-2">
            <Card className="border-border bg-card">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-foreground flex items-center gap-2">
                    <Clock className="h-5 w-5 text-primary" />
                    Generation History
                  </CardTitle>
                  <CardDescription className="text-muted-foreground">
                    Your past market scenario configurations
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={fetchHistory}
                  disabled={loadingHistory}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <RefreshCw className={`h-4 w-4 ${loadingHistory ? "animate-spin" : ""}`} />
                </Button>
              </CardHeader>
              <CardContent>
                {loadingHistory ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : history.length === 0 ? (
                  <div className="text-center py-12">
                    <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-foreground mb-2">No history yet</h3>
                    <p className="text-muted-foreground mb-4">
                      Your generated market scenarios will appear here
                    </p>
                    <Link href="/">
                      <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                        Create Your First Scenario
                      </Button>
                    </Link>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {history.map((record) => (
                      <div
                        key={record.id}
                        className="p-4 rounded-lg border border-border bg-secondary/30 hover:bg-secondary/50 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-2">
                              <h4 className="font-medium text-foreground truncate">
                                {record.scenario_name}
                              </h4>
                              {trendIcons[record.trend_type]}
                            </div>

                            <div className="flex flex-wrap items-center gap-2 mb-3">
                              <Badge
                                variant="outline"
                                className="border-border text-muted-foreground"
                              >
                                {formatLabel(record.asset_class)}
                              </Badge>
                              <Badge
                                variant="outline"
                                className={regimeColors[record.market_regime] || "border-border"}
                              >
                                {formatLabel(record.market_regime)}
                              </Badge>
                              <Badge
                                variant="outline"
                                className="border-border text-muted-foreground"
                              >
                                {formatLabel(record.volatility_level)}
                              </Badge>
                            </div>

                            <div className="flex items-center gap-4 text-xs text-muted-foreground">
                              <span>{record.data_points.toLocaleString()} data points</span>
                              <span>{record.time_horizon} days</span>
                              <span className="uppercase">{record.generation_model}</span>
                            </div>

                            <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              {formatDate(record.created_at)}
                            </div>
                          </div>

                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDelete(record.id)}
                            disabled={deletingId === record.id}
                            className="text-muted-foreground hover:text-destructive shrink-0"
                          >
                            {deletingId === record.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
