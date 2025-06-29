import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from "recharts";
import { TrendingDown, BarChart2, Layers } from "lucide-react";

// Mock data for charts
const lossData = Array.from({ length: 30 }, (_, i) => ({
  step: i * 50,
  trainingLoss: 4.0 - (i * 0.05) + (Math.random() * 0.08 - 0.04),
  validationLoss: 4.1 - (i * 0.045) + (Math.random() * 0.12 - 0.06),
}));

const expertData = [
  { expert: 'Expert 1', utilization: 23, efficiency: 95 },
  { expert: 'Expert 2', utilization: 18, efficiency: 92 },
  { expert: 'Expert 3', utilization: 15, efficiency: 88 },
  { expert: 'Expert 4', utilization: 22, efficiency: 94 },
  { expert: 'Expert 5', utilization: 19, efficiency: 91 },
  { expert: 'Expert 6', utilization: 16, efficiency: 89 },
  { expert: 'Expert 7', utilization: 8, efficiency: 75 },
  { expert: 'Expert 8', utilization: 4, efficiency: 68 },
];

const architectureData = [
  { layer: 'Attention', active: 80, total: 100 },
  { layer: 'MoE', active: 40, total: 100 },
  { layer: 'MoD Enabled', active: 60, total: 100 },
];

export function DetailedCharts() {
  return (
    <div className="space-y-6">
      {/* Loss Curves and Expert Utilization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss Curves */}
        <Card className="overflow-hidden">
          <CardHeader className="bg-muted/50">
            <CardTitle className="flex items-center">
              <TrendingDown className="w-5 h-5 mr-2 text-blue-500" />
              Loss Curves
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey="step" 
                    stroke="hsl(var(--muted-foreground))"
                  />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'hsl(var(--popover))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                      color: 'hsl(var(--popover-foreground))'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="trainingLoss" 
                    stroke="hsl(var(--chart-1))" 
                    strokeWidth={2}
                    dot={false}
                    name="Training Loss"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="validationLoss" 
                    stroke="hsl(var(--chart-2))" 
                    strokeWidth={2}
                    dot={false}
                    name="Validation Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        
        {/* Expert Utilization */}
        <Card className="overflow-hidden">
          <CardHeader className="bg-muted/50">
            <CardTitle className="flex items-center">
              <BarChart2 className="w-5 h-5 mr-2 text-purple-500" />
              Expert Utilization
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={expertData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis 
                    dataKey="expert" 
                    stroke="hsl(var(--muted-foreground))"
                  />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'hsl(var(--popover))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                      color: 'hsl(var(--popover-foreground))'
                    }}
                  />
                  <Bar 
                    dataKey="utilization" 
                    fill="hsl(var(--chart-3))"
                    name="Utilization %"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Architecture Visualization */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <Layers className="w-5 h-5 mr-2 text-blue-500" />
            Model Architecture
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Architecture Overview */}
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Architecture Overview</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <span className="text-sm text-muted-foreground">Total Parameters</span>
                  <span className="text-sm font-medium text-foreground">7.2B</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <span className="text-sm text-muted-foreground">Active Parameters</span>
                  <span className="text-sm font-medium text-foreground">1.8B</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <span className="text-sm text-muted-foreground">Hidden Dimensions</span>
                  <span className="text-sm font-medium text-foreground">4096</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                  <span className="text-sm text-muted-foreground">Attention Heads</span>
                  <span className="text-sm font-medium text-foreground">32</span>
                </div>
              </div>
            </div>
            
            {/* Layer Statistics */}
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Layer Statistics</h4>
              <div className="space-y-4">
                {architectureData.map((item, index) => (
                  <div key={item.layer} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{item.layer}</span>
                      <span className="text-foreground font-medium">{item.active}</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          index === 0 ? 'bg-blue-500' : 
                          index === 1 ? 'bg-purple-500' : 
                          'bg-green-500'
                        }`}
                        style={{ width: `${(item.active / item.total) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
