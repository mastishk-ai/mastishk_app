import { Card, CardContent } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  iconColor?: string;
  trend?: {
    value: string;
    direction: 'up' | 'down' | 'neutral';
  };
  className?: string;
}

export function MetricCard({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  iconColor = "text-blue-500",
  trend,
  className 
}: MetricCardProps) {
  const getTrendColor = (direction: string) => {
    switch (direction) {
      case 'up': return 'text-green-500';
      case 'down': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'up': return '↑';
      case 'down': return '↓';
      default: return '→';
    }
  };

  return (
    <Card className={`metric-card ${className}`}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold text-foreground">{value}</p>
          </div>
          <div className={`w-10 h-10 bg-opacity-20 rounded-lg flex items-center justify-center ${iconColor.replace('text-', 'bg-')}`}>
            <Icon className={`w-5 h-5 ${iconColor}`} />
          </div>
        </div>
        
        {(subtitle || trend) && (
          <div className="mt-4 flex items-center text-sm">
            {trend && (
              <span className={getTrendColor(trend.direction)}>
                {getTrendIcon(trend.direction)} {trend.value}
              </span>
            )}
            {subtitle && (
              <span className={`text-muted-foreground ${trend ? 'ml-2' : ''}`}>
                {subtitle}
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
