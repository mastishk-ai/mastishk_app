import { MetricCard } from "@/components/ui/metric-card";
import { TrendingUp, Zap, GitBranch } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

interface PerformanceMetrics {
  perplexity: number;
  bleuScore: number;
  tokensPerSecond: number;
  gpuUtilization: number;
  memoryEfficiency: number;
  flopsUtilization: number;
  expertLoadBalance: number;
  routerEfficiency: number;
  activeExperts: string;
}

export function PerformanceOverview() {
  // This would fetch actual performance metrics
  const { data: metrics } = useQuery({
    queryKey: ['/api/analytics/performance'],
    initialData: {
      perplexity: 15.7,
      bleuScore: 0.847,
      tokensPerSecond: 1247,
      gpuUtilization: 87.3,
      memoryEfficiency: 92.1,
      flopsUtilization: 78.9,
      expertLoadBalance: 0.89,
      routerEfficiency: 94.2,
      activeExperts: "6/8"
    } as PerformanceMetrics
  });

  if (!metrics) return null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Model Performance */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-foreground flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-emerald-500" />
          Model Performance
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">Perplexity</span>
            <span className="text-sm font-medium text-foreground">{metrics.perplexity}</span>
          </div>
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">BLEU Score</span>
            <span className="text-sm font-medium text-foreground">{metrics.bleuScore}</span>
          </div>
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">Tokens/sec</span>
            <span className="text-sm font-medium text-foreground">{metrics.tokensPerSecond.toLocaleString()}</span>
          </div>
        </div>
      </div>
      
      {/* Training Efficiency */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-foreground flex items-center">
          <Zap className="w-5 h-5 mr-2 text-blue-500" />
          Training Efficiency
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">GPU Utilization</span>
            <span className="text-sm font-medium text-foreground">{metrics.gpuUtilization}%</span>
          </div>
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">Memory Efficiency</span>
            <span className="text-sm font-medium text-foreground">{metrics.memoryEfficiency}%</span>
          </div>
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">FLOPs Utilization</span>
            <span className="text-sm font-medium text-foreground">{metrics.flopsUtilization}%</span>
          </div>
        </div>
      </div>
      
      {/* MoE Statistics */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-foreground flex items-center">
          <GitBranch className="w-5 h-5 mr-2 text-purple-500" />
          MoE Statistics
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">Expert Load Balance</span>
            <span className="text-sm font-medium text-foreground">{metrics.expertLoadBalance}</span>
          </div>
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">Router Efficiency</span>
            <span className="text-sm font-medium text-foreground">{metrics.routerEfficiency}%</span>
          </div>
          <div className="flex justify-between p-3 bg-muted rounded-lg">
            <span className="text-sm text-muted-foreground">Active Experts</span>
            <span className="text-sm font-medium text-foreground">{metrics.activeExperts}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
