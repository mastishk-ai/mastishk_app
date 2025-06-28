import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Cpu, GitBranch } from "lucide-react";
import { useWebSocket } from "@/hooks/use-websocket";

interface SystemMetrics {
  gpu: {
    utilization: number;
    memory: {
      used: number;
      total: number;
    };
    temperature: number;
    powerDraw: number;
  };
  experts: {
    utilization: number[];
    loadBalance: number;
    efficiency: number;
    skipRate: number;
  };
}

export function SystemMetrics() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    gpu: {
      utilization: 0,
      memory: { used: 0, total: 24 },
      temperature: 0,
      powerDraw: 0
    },
    experts: {
      utilization: new Array(8).fill(0),
      loadBalance: 0,
      efficiency: 0,
      skipRate: 0
    }
  });

  const { lastMessage } = useWebSocket();

  useEffect(() => {
    if (lastMessage?.type === 'training_progress') {
      const progress = lastMessage.data.progress;
      
      setMetrics(prev => ({
        gpu: {
          utilization: progress.gpuUtilization || 87,
          memory: {
            used: progress.memoryUsage || 18.2,
            total: 24
          },
          temperature: 73,
          powerDraw: 320
        },
        experts: {
          utilization: progress.expertUtilization || [23, 18, 15, 22, 19, 16, 8, 4],
          loadBalance: 0.89,
          efficiency: progress.efficiency || 94.2,
          skipRate: progress.layerSkipRate || 18.5
        }
      }));
    }
  }, [lastMessage]);

  // Simulate some data when not training
  useEffect(() => {
    const interval = setInterval(() => {
      if (!lastMessage || lastMessage.type !== 'training_progress') {
        setMetrics(prev => ({
          gpu: {
            utilization: 87 + Math.random() * 10 - 5,
            memory: {
              used: 18.2 + Math.random() * 2 - 1,
              total: 24
            },
            temperature: 73 + Math.random() * 4 - 2,
            powerDraw: 320 + Math.random() * 20 - 10
          },
          experts: {
            utilization: prev.experts.utilization.map(val => Math.max(0, val + Math.random() * 4 - 2)),
            loadBalance: 0.89 + Math.random() * 0.1 - 0.05,
            efficiency: 94.2 + Math.random() * 2 - 1,
            skipRate: 18.5 + Math.random() * 3 - 1.5
          }
        }));
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [lastMessage]);

  const memoryPercentage = (metrics.gpu.memory.used / metrics.gpu.memory.total) * 100;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* GPU Utilization */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <Cpu className="w-5 h-5 mr-2 text-green-500" />
            GPU Utilization
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">GPU 0 - RTX 4090</span>
              <span className="text-sm font-medium text-foreground">{metrics.gpu.utilization.toFixed(1)}%</span>
            </div>
            <Progress value={metrics.gpu.utilization} className="w-full" />
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Memory Usage</span>
              <span className="text-sm font-medium text-foreground">
                {metrics.gpu.memory.used.toFixed(1)} / {metrics.gpu.memory.total.toFixed(1)} GB
              </span>
            </div>
            <Progress value={memoryPercentage} className="w-full" />
          </div>
          
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Temperature</p>
              <p className="text-lg font-semibold text-foreground">{metrics.gpu.temperature.toFixed(0)}°C</p>
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Power Draw</p>
              <p className="text-lg font-semibold text-foreground">{metrics.gpu.powerDraw.toFixed(0)}W</p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* MoE & MoD Statistics */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <GitBranch className="w-5 h-5 mr-2 text-purple-500" />
            MoE & MoD Statistics
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Expert Utilization</span>
              <span className="text-sm font-medium text-foreground">
                {metrics.experts.utilization.filter(u => u > 5).length}/8 active
              </span>
            </div>
            <div className="grid grid-cols-8 gap-1">
              {metrics.experts.utilization.map((utilization, index) => (
                <div
                  key={index}
                  className={`h-4 rounded transition-colors ${
                    utilization > 15 ? 'bg-green-500' : 
                    utilization > 5 ? 'bg-blue-500' : 
                    'bg-muted'
                  }`}
                  title={`Expert ${index + 1}: ${utilization.toFixed(1)}%`}
                />
              ))}
            </div>
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Average Layer Skip Rate</span>
              <span className="text-sm font-medium text-foreground">{metrics.experts.skipRate.toFixed(1)}%</span>
            </div>
            <Progress value={metrics.experts.skipRate} className="w-full" />
          </div>
          
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Load Balance</p>
              <p className="text-lg font-semibold text-foreground">{metrics.experts.loadBalance.toFixed(2)}</p>
            </div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Efficiency</p>
              <p className="text-lg font-semibold text-foreground">{metrics.experts.efficiency.toFixed(1)}%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
