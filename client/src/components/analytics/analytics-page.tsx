import { PerformanceOverview } from "./performance-overview";
import { DetailedCharts } from "./detailed-charts";

export function AnalyticsPageContent() {
  return (
    <div className="space-y-6">
      {/* Performance Overview */}
      <PerformanceOverview />
      
      {/* Detailed Charts */}
      <DetailedCharts />
    </div>
  );
}

// Export as AnalyticsPage for consistency
export { AnalyticsPageContent as AnalyticsPage };
