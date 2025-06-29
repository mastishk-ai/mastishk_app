import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Activity, TrendingUp, Clock, Zap } from 'lucide-react';

export function TrainingMonitor() {
  return (
    <div className="space-y-6 p-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg text-truncate">
              <Activity className="w-5 h-5 text-primary flex-shrink-0" />
              <span className="text-truncate">Training Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="card-content-spacing">
            <div className="flex items-center gap-2 overflow-hidden">
              <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 flex-shrink-0">
                Ready
              </Badge>
              <span className="text-sm text-muted-foreground text-truncate">No active training</span>
            </div>
          </CardContent>
        </Card>

        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg text-truncate">
              <TrendingUp className="w-5 h-5 text-primary flex-shrink-0" />
              <span className="text-truncate">Training Loss</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="card-content-spacing">
            <div className="text-2xl font-bold text-primary">--</div>
            <p className="text-sm text-muted-foreground break-words">No recent training data</p>
          </CardContent>
        </Card>

        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg text-truncate">
              <Clock className="w-5 h-5 text-primary flex-shrink-0" />
              <span className="text-truncate">Duration</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="card-content-spacing">
            <div className="text-2xl font-bold text-primary">00:00:00</div>
            <p className="text-sm text-muted-foreground break-words">Training time</p>
          </CardContent>
        </Card>
      </div>

      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-truncate">
            <Zap className="w-5 h-5 text-primary flex-shrink-0" />
            <span className="text-truncate">Training Visualization</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="card-content-spacing">
          <div className="h-[300px] lg:h-[400px] flex items-center justify-center rounded-xl bg-gradient-to-br from-primary/5 to-primary/10 overflow-hidden">
            <div className="text-center p-6 content-container">
              <div className="w-12 h-12 lg:w-16 lg:h-16 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4">
                <Activity className="w-6 h-6 lg:w-8 lg:h-8 text-primary" />
              </div>
              <h3 className="text-base lg:text-lg font-semibold text-foreground mb-2 break-words">Training Monitor</h3>
              <p className="text-muted-foreground text-sm lg:text-base break-words">Start a training session to see real-time metrics and visualizations</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}