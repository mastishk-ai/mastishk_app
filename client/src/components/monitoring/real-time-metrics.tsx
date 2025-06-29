import { useEffect, useState } from "react";
import { MetricCard } from "@/components/ui/metric-card";
import { WeightVerificationPanel } from "@/components/training/weight-verification-panel";
import { TrendingDown, Zap, CheckCircle, Clock } from "lucide-react";
import { useWebSocket } from "@/hooks/use-websocket";
import { useTraining } from "@/hooks/use-training";

interface Metrics {
  currentLoss: number;
  learningRate: number;
  stepsCompleted: number;
  eta: string;
  trend: {
    loss: number;
    direction: 'up' | 'down' | 'neutral';
  };
}

interface WeightVerificationData {
  step: number;
  weights_updated: boolean;
  layers_changed: number;
  verification_status: string;
  optimizer_stepped: boolean;
  snapshot_id?: string;
}

export function RealTimeMetrics() {
  const [metrics, setMetrics] = useState<Metrics>({
    currentLoss: 0,
    learningRate: 0,
    stepsCompleted: 0,
    eta: "0h 0m",
    trend: { loss: 0, direction: 'neutral' }
  });
  
  const [weightVerificationData, setWeightVerificationData] = useState<WeightVerificationData | null>(null);
  
  const { lastMessage } = useWebSocket();
  const { activeRun, trainingStatus, config } = useTraining();

  useEffect(() => {
    if (lastMessage?.type === 'training_progress') {
      const progress = lastMessage.data.progress;
      const prevLoss = metrics.currentLoss;
      
      setMetrics({
        currentLoss: progress.loss || 0,
        learningRate: progress.learningRate || 0,
        stepsCompleted: progress.step || 0,
        eta: progress.eta || "Calculating...",
        trend: {
          loss: prevLoss > 0 ? ((progress.loss - prevLoss) / prevLoss * 100) : 0,
          direction: prevLoss > 0 ? (progress.loss < prevLoss ? 'down' : 'up') : 'neutral'
        }
      });

      // Update weight verification data if available in progress
      if (progress.weights_updated !== undefined) {
        setWeightVerificationData({
          step: progress.step || 0,
          weights_updated: progress.weights_updated || false,
          layers_changed: progress.layers_changed || 0,
          verification_status: progress.verification_status || 'Unknown',
          optimizer_stepped: progress.optimizer_stepped || false,
          snapshot_id: progress.snapshot_id
        });
      }
    }

    // Handle dedicated weight snapshot messages
    if (lastMessage?.type === 'weight_snapshot') {
      const weightData = lastMessage.data;
      setWeightVerificationData({
        step: weightData.step || 0,
        weights_updated: weightData.weights_updated || false,
        layers_changed: weightData.layers_changed || 0,
        verification_status: weightData.verification_status || 'Unknown',
        optimizer_stepped: weightData.optimizer_stepped || false,
        snapshot_id: weightData.snapshot_id
      });
    }
  }, [lastMessage, metrics.currentLoss]);

  // Update from active run if available
  useEffect(() => {
    if (activeRun) {
      setMetrics(prev => ({
        ...prev,
        currentLoss: activeRun.currentLoss || 0,
        learningRate: activeRun.learningRate || 0,
        stepsCompleted: activeRun.currentStep || 0,
      }));
    }
  }, [activeRun]);

  const formatLoss = (loss: number) => {
    if (loss === 0) return "0.000";
    return loss.toFixed(3);
  };

  const formatLearningRate = (lr: number) => {
    if (lr === 0) return "0.0e-0";
    return lr.toExponential(1);
  };

  const formatSteps = (steps: number) => {
    if (steps >= 1000) {
      return `${(steps / 1000).toFixed(1)}k`;
    }
    return steps.toString();
  };

  const formatTrend = (trend: number, direction: string) => {
    const prefix = direction === 'down' ? '↓' : direction === 'up' ? '↑' : '→';
    return `${prefix} ${Math.abs(trend).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Core Training Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Current Loss"
          value={formatLoss(metrics.currentLoss)}
          icon={TrendingDown}
          iconColor="text-blue-500"
          trend={{
            value: formatTrend(metrics.trend.loss, metrics.trend.direction),
            direction: metrics.trend.direction
          }}
          subtitle="vs last epoch"
        />
        
        <MetricCard
          title="Learning Rate"
          value={formatLearningRate(metrics.learningRate)}
          icon={Zap}
          iconColor="text-purple-500"
          subtitle="Cosine schedule"
        />
        
        <MetricCard
          title="Steps Completed"
          value={formatSteps(metrics.stepsCompleted)}
          icon={CheckCircle}
          iconColor="text-emerald-500"
          subtitle={activeRun ? `of ${formatSteps(activeRun.totalSteps)} total` : "No active training"}
        />
        
        <MetricCard
          title="ETA"
          value={metrics.eta}
          icon={Clock}
          iconColor="text-amber-500"
          subtitle="Estimated completion"
        />
      </div>

      {/* Weight Verification Panel - Only show when weight logging is enabled */}
      {config.enable_weight_logging && (
        <WeightVerificationPanel 
          data={weightVerificationData}
          enabled={config.enable_weight_logging}
          totalLayers={32}
        />
      )}
    </div>
  );
}
