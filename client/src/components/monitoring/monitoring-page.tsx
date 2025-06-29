import { RealTimeMetrics } from "./real-time-metrics";
import { TrainingCharts } from "./training-charts";
import { SystemMetrics } from "./system-metrics";
import { TrainingLogs } from "./training-logs";

export function MonitoringPage() {
  return (
    <div className="space-y-6">
      {/* Real-time Metrics */}
      <RealTimeMetrics />
      
      {/* Training Loss Chart */}
      <TrainingCharts />
      
      {/* System Metrics */}
      <SystemMetrics />
      
      {/* Training Logs */}
      <TrainingLogs />
    </div>
  );
}
