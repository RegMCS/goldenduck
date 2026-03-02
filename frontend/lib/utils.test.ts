import { describe, expect, it } from "vitest"
import { cn } from "@/lib/utils"

describe("cn", () => {
  it("merges tailwind classes and resolves conflicts", () => {
    expect(cn("px-2", "text-sm", "px-4")).toBe("text-sm px-4")
  })

  it("filters out falsy values", () => {
    expect(cn("font-semibold", false && "hidden", null, undefined)).toBe("font-semibold")
  })
})
