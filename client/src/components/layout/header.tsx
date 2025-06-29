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
        
        <div className="flex items-center space-x-4">
          {/* Training Status Indicator */}
          <div className="training-status-indicator">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(
              trainingStatus?.isTraining || false, 
              trainingStatus?.status || 'idle'
            )}`}></div>
            <span className="text-xs font-medium text-foreground">
              {getStatusText(trainingStatus?.isTraining || false, trainingStatus?.status || 'idle')}
            </span>
          </div>
          
          {/* Global Actions */}
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm">
              <Download className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <Upload className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <HelpCircle className="w-4 h-4" />
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
