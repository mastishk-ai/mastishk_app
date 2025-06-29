import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { TrendingDown } from "lucide-react";
import { useTraining } from "@/hooks/use-training";
import { useWebSocket } from "@/hooks/use-websocket";

interface MetricPoint {
  step: number;
  loss: number;
  learningRate: number;
  timestamp: string;
}

export function TrainingCharts() {
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const { activeRun } = useTraining();
  const { lastMessage } = useWebSocket();

  // Load existing metrics when component mounts
  useEffect(() => {
    if (activeRun) {
      // This would fetch historical metrics
      // For now, we'll simulate some data
      const simulatedMetrics: MetricPoint[] = [];
      for (let i = 0; i < 50; i++) {
        simulatedMetrics.push({
          step: i * 20,
          loss: 4.5 - (i * 0.03) + (Math.random() * 0.1 - 0.05),
          learningRate: 5e-4 * Math.cos((i / 50) * Math.PI * 0.5),
          timestamp: new Date(Date.now() - (50 - i) * 60000).toISOString()
        });
      }
      setMetrics(simulatedMetrics);
    }
  }, [activeRun]);

  // Update metrics from WebSocket
  useEffect(() => {
    if (lastMessage?.type === 'training_progress') {
      const progress = lastMessage.data.progress;
      const newPoint: MetricPoint = {
        step: progress.step,
        loss: progress.loss,
        learningRate: progress.learningRate,
        timestamp: new Date().toISOString()
      };
      
      setMetrics(prev => {
        const updated = [...prev, newPoint];
        // Keep only last 100 points for performance
        return updated.slice(-100);
      });
    }
  }, [lastMessage]);

  const formatTooltipValue = (value: number, name: string) => {
    if (name === 'learningRate') {
      return [value.toExponential(2), 'Learning Rate'];
    }
    if (name === 'loss') {
      return [value.toFixed(4), 'Loss'];
    }
    return [value, name];
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <TrendingDown className="w-5 h-5 mr-2 text-blue-500" />
          Training Loss
        </CardTitle>
        <p className="text-sm text-muted-foreground">Real-time training metrics visualization</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="h-80">
          {metrics.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="step" 
                  stroke="hsl(var(--muted-foreground))"
                  label={{ value: 'Training Steps', position: 'insideBottom', offset: -10 }}
                />
                <YAxis 
                  yAxisId="loss"
                  stroke="hsl(var(--muted-foreground))"
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                />
                <YAxis 
                  yAxisId="lr"
                  orientation="right"
                  stroke="hsl(var(--muted-foreground))"
                  label={{ value: 'Learning Rate', angle: 90, position: 'insideRight' }}
                />
                <Tooltip 
                  formatter={formatTooltipValue}
                  labelFormatter={(step) => `Step: ${step}`}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                    color: 'hsl(var(--popover-foreground))'
                  }}
                />
                <Legend />
                <Line 
                  yAxisId="loss"
                  type="monotone" 
                  dataKey="loss" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  dot={false}
                  name="Training Loss"
                />
                <Line 
                  yAxisId="lr"
                  type="monotone" 
                  dataKey="learningRate" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  dot={false}
                  name="Learning Rate"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <div className="text-center">
                <TrendingDown className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No training data available</p>
                <p className="text-sm">Start training to see real-time metrics</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
