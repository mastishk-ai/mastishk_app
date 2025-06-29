import { Moon, Sun, Monitor } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useState, useEffect } from 'react';

export function ThemeToggle() {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system');

  useEffect(() => {
    // Check initial theme
    const storedTheme = (localStorage.getItem('theme') as 'light' | 'dark' | 'system') || 'system';
    setTheme(storedTheme);
    applyTheme(storedTheme);
  }, []);

  const applyTheme = (newTheme: 'light' | 'dark' | 'system') => {
    const isSystemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldBeDark = newTheme === 'dark' || (newTheme === 'system' && isSystemDark);
    
    // Force remove and add the dark class to ensure it applies
    document.documentElement.classList.remove('dark');
    if (shouldBeDark) {
      document.documentElement.classList.add('dark');
    }
    
    console.log('Applied theme:', { newTheme, shouldBeDark, classList: document.documentElement.classList.toString() });
  };

  const changeTheme = (newTheme: 'light' | 'dark' | 'system') => {
    console.log('Changing theme to:', newTheme);
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    applyTheme(newTheme);
    
    // Trigger a custom event to notify other components
    window.dispatchEvent(new CustomEvent('themeChange', { detail: newTheme }));
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="w-10 h-10 rounded-xl premium-card hover:premium-button transition-all duration-300">
          <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52 premium-card p-2">
        <DropdownMenuItem 
          onClick={() => changeTheme('light')} 
          className="cursor-pointer rounded-lg px-3 py-2.5 font-semibold hover:bg-primary/10 transition-colors"
        >
          <Sun className="mr-3 h-5 w-5" />
          <span>Light Mode</span>
        </DropdownMenuItem>
        <DropdownMenuItem 
          onClick={() => changeTheme('dark')} 
          className="cursor-pointer rounded-lg px-3 py-2.5 font-semibold hover:bg-primary/10 transition-colors"
        >
          <Moon className="mr-3 h-5 w-5" />
          <span>Dark Mode</span>
        </DropdownMenuItem>
        <DropdownMenuItem 
          onClick={() => changeTheme('system')} 
          className="cursor-pointer rounded-lg px-3 py-2.5 font-semibold hover:bg-primary/10 transition-colors"
        >
          <Monitor className="mr-3 h-5 w-5" />
          <span>System</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}