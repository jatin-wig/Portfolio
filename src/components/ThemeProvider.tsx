import { ThemeProvider as NextThemesProvider } from "next-themes"
import { type ThemeProviderProps } from "next-themes/dist/types"

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return <NextThemesProvider {...props} defaultTheme="dark" forcedTheme="dark" enableSystem={false}>{children}</NextThemesProvider>
}

// Re-export useTheme for convenience if needed, 
// though direct import from next-themes is also fine.
export { useTheme } from "next-themes"
