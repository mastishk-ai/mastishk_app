import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { Download, Upload, HelpCircle } from "lucide-react";

interface HeaderProps {
  title: string;
  subtitle: string;
  trainingStatus?: {
    isTraining: boolean;
    status: string;
  };
}

export function Header({ title, subtitle, trainingStatus }: HeaderProps) {
  const getStatusColor = (isTraining: boolean, status: string) => {
    if (isTraining) return 'bg-blue-500';
    switch (status) {
      case 'completed': return 'bg-emerald-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (isTraining: boolean, status: string) => {
    if (isTraining) return 'Training';
    switch (status) {
      case 'completed': return 'Completed';
      case 'error': return 'Error';
      default: return 'Idle';
    }
  };

  return (
    <header className="glass-effect border-b px-8 py-6 animate-fade-in-up">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent">
            {title}
          </h1>
          <p className="text-muted-foreground font-medium">{subtitle}</p>
        </div>
        
        <div className="flex items-center space-x-6">
          {/* Training Status Indicator */}
          <div className="premium-card px-4 py-2 rounded-xl">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                trainingStatus?.isTraining 
                  ? 'bg-emerald-500 shadow-lg shadow-emerald-500/50 animate-pulse' 
                  : 'bg-slate-400'
              }`} />
              <span className="text-sm font-semibold">
                {getStatusText(trainingStatus?.isTraining || false, trainingStatus?.status || 'idle')}
              </span>
            </div>
          </div>
          
          {/* Global Actions */}
          <div className="flex items-center space-x-3">
            <Button variant="ghost" size="sm" className="premium-button bg-transparent hover:bg-primary/10 rounded-xl">
              <Download className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="sm" className="premium-button bg-transparent hover:bg-primary/10 rounded-xl">
              <Upload className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="sm" className="premium-button bg-transparent hover:bg-primary/10 rounded-xl">
              <HelpCircle className="w-5 h-5" />
            </Button>
            <div className="h-8 w-px bg-border/50" />
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
