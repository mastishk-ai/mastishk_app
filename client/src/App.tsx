import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useEffect, useState } from "react";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import { VisualizationPage } from "@/pages/visualization";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/training" component={Dashboard} />
      <Route path="/generation" component={Dashboard} />
      <Route path="/monitoring" component={Dashboard} />
      <Route path="/checkpoints" component={Dashboard} />
      <Route path="/analytics" component={Dashboard} />
      <Route path="/testing" component={Dashboard} />
      <Route path="/docs" component={Dashboard} />
      <Route path="/visualization" component={VisualizationPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Initialize theme on app start
    const initializeTheme = () => {
      const theme = localStorage.getItem('theme') || 'system';
      const isSystemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const shouldBeDark = theme === 'dark' || (theme === 'system' && isSystemDark);
      
      document.documentElement.classList.toggle('dark', shouldBeDark);
      console.log('Theme initialized:', { theme, isSystemDark, shouldBeDark });
    };

    initializeTheme();
    setMounted(true);

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleSystemChange = () => {
      if (localStorage.getItem('theme') === 'system') {
        const shouldBeDark = mediaQuery.matches;
        document.documentElement.classList.toggle('dark', shouldBeDark);
        console.log('System theme changed:', shouldBeDark);
      }
    };

    // Listen for custom theme change events
    const handleThemeChange = (event: CustomEvent) => {
      const newTheme = event.detail;
      const isSystemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const shouldBeDark = newTheme === 'dark' || (newTheme === 'system' && isSystemDark);
      
      document.documentElement.classList.toggle('dark', shouldBeDark);
      console.log('Theme changed:', { newTheme, shouldBeDark });
    };

    mediaQuery.addEventListener('change', handleSystemChange);
    window.addEventListener('themeChange', handleThemeChange as EventListener);
    
    return () => {
      mediaQuery.removeEventListener('change', handleSystemChange);
      window.removeEventListener('themeChange', handleThemeChange as EventListener);
    };
  }, []);

  if (!mounted) {
    return null; // Prevent flash of unstyled content
  }

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div className="min-h-screen bg-background text-foreground transition-colors duration-200" style={{backgroundColor: 'hsl(var(--background))', color: 'hsl(var(--foreground))'}}>
          <Toaster />
          <Router />
        </div>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
