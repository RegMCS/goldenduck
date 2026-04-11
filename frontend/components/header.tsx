"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { Activity, Sun, Moon, History, LogOut, User, BrainCircuit, Sliders, Shield } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/components/auth-provider"


import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function Header() {
  const { theme, setTheme } = useTheme()
  const pathname = usePathname()
  const { user, loading, logout } = useAuth()
  const router = useRouter()

  return (
    <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary shadow-lg shadow-primary/20">
              <Activity className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-foreground tracking-tight">SynthMarket</h1>
              <p className="text-xs text-muted-foreground">Synthetic Data Generator</p>
            </div>
          </Link>
          {/* Nav links */}
          <nav className="hidden sm:flex items-center gap-0.5 ml-4">
            <Button
              variant="ghost"
              size="sm"
              className={`gap-1.5 ${pathname === "/"
                ? "text-foreground font-semibold bg-muted"
                : "text-muted-foreground hover:text-foreground"
                }`}
              asChild
            >
              <Link href="/">
                <Sliders className="h-4 w-4" />
                Configure
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className={`gap-1.5 ${pathname === "/reports"
                ? "text-foreground font-semibold bg-muted"
                : "text-muted-foreground hover:text-foreground"
                }`}
              asChild
              id="nav-reports-link"
            >
              <Link href="/reports">
                <History className="h-4 w-4" />
                Reports
              </Link>
            </Button>
            {user?.is_admin && (
              <Button
                variant="ghost"
                size="sm"
                className={`gap-1.5 ${pathname === "/models"
                  ? "text-foreground font-semibold bg-muted"
                  : "text-muted-foreground hover:text-foreground"
                  }`}
                asChild
                id="nav-models-link"
              >
                <Link href="/models">
                  <BrainCircuit className="h-4 w-4" />
                  Models
                </Link>
              </Button>
            )}
            {user?.is_admin && (
              <Button
                variant="ghost"
                size="sm"
                className={`gap-1.5 ${pathname === "/admin/users"
                  ? "text-foreground font-semibold bg-muted"
                  : "text-muted-foreground hover:text-foreground"
                  }`}
                asChild
              >
                <Link href="/admin/users">
                  <Shield className="h-4 w-4" />
                  Users
                </Link>
              </Button>
            )}
          </nav>

        </div>
        <div className="flex items-center gap-1">
          {!loading && (
            <>
              {user ? (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
                      <User className="h-4 w-4" />
                      <span className="hidden sm:inline">{user.username}</span>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    {/* <DropdownMenuItem asChild>
                      <Link href="/profile">Profile</Link>
                    </DropdownMenuItem> */}
                    <DropdownMenuItem onClick={async () => { await logout(); router.push("/login") }}>
                      <LogOut className="h-4 w-4" />
                      Log out
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              ) : (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground hover:text-foreground"
                    asChild
                  >
                    <Link href="/login">Log in</Link>
                  </Button>
                  <Button size="sm" asChild>
                    <Link href="/register" prefetch={false}>Register</Link>
                  </Button>
                </>
              )}
            </>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground hover:text-foreground"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
