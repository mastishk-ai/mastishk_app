import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Activity, CheckCircle, XCircle, Eye, AlertTriangle } from "lucide-react";

interface WeightVerificationData {
  step: number;
  weights_updated: boolean;
  layers_changed: number;
  verification_status: string;
  optimizer_stepped: boolean;
  snapshot_id?: string;
}

interface WeightVerificationPanelProps {
  data: WeightVerificationData | null;
  enabled: boolean;
  totalLayers?: number;
}

export function WeightVerificationPanel({ data, enabled, totalLayers = 32 }: WeightVerificationPanelProps) {
  if (!enabled) {
    return (
      <Card className="border-gray-200 dark:border-gray-800">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Eye className="w-4 h-4 text-gray-400" />
            Weight Verification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-6 text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-gray-400" />
              <p className="text-sm">Weight logging disabled</p>
              <p className="text-xs text-gray-400">Enable in training configuration</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="border-blue-200 dark:border-blue-800">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
            Weight Verification
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-6 text-blue-500 dark:text-blue-400">
            <div className="text-center">
              <Activity className="w-8 h-8 mx-auto mb-2 animate-pulse" />
              <p className="text-sm">Waiting for weight data...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const layersChangedPercentage = (data.layers_changed / totalLayers) * 100;
  const isHealthy = data.weights_updated && data.layers_changed > 0;

  return (
    <Card className={`border-2 ${isHealthy ? 'border-green-200 dark:border-green-800' : 'border-red-200 dark:border-red-800'}`}>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          {isHealthy ? (
            <CheckCircle className="w-4 h-4 text-green-500" />
          ) : (
            <XCircle className="w-4 h-4 text-red-500" />
          )}
          Weight Verification
          <Badge variant={isHealthy ? "default" : "destructive"} className="ml-auto">
            Step {data.step}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Verification Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Status</span>
          <span className={`text-xs px-2 py-1 rounded-full ${
            isHealthy 
              ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
              : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          }`}>
            {data.verification_status}
          </span>
        </div>

        {/* Weight Update Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Weights Updated</span>
          <div className="flex items-center gap-2">
            {data.weights_updated ? (
              <CheckCircle className="w-4 h-4 text-green-500" />
            ) : (
              <XCircle className="w-4 h-4 text-red-500" />
            )}
            <span className="text-sm">{data.weights_updated ? 'Yes' : 'No'}</span>
          </div>
        </div>

        {/* Optimizer Step Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Optimizer Stepped</span>
          <div className="flex items-center gap-2">
            {data.optimizer_stepped ? (
              <CheckCircle className="w-4 h-4 text-green-500" />
            ) : (
              <XCircle className="w-4 h-4 text-orange-500" />
            )}
            <span className="text-sm">{data.optimizer_stepped ? 'Yes' : 'Accumulating'}</span>
          </div>
        </div>

        {/* Layers Changed */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Layers Changed</span>
            <span className="text-sm font-mono">
              {data.layers_changed}/{totalLayers} ({layersChangedPercentage.toFixed(1)}%)
            </span>
          </div>
          <Progress 
            value={layersChangedPercentage} 
            className={`h-2 ${isHealthy ? 'bg-green-100 dark:bg-green-900/20' : 'bg-red-100 dark:bg-red-900/20'}`}
          />
        </div>

        {/* Snapshot ID */}
        {data.snapshot_id && (
          <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Snapshot ID</span>
              <span className="text-xs font-mono text-gray-600 dark:text-gray-400">
                {data.snapshot_id}
              </span>
            </div>
          </div>
        )}

        {/* Health Indicator */}
        <div className={`p-3 rounded-lg ${
          isHealthy 
            ? 'bg-green-50 border border-green-200 dark:bg-green-900/10 dark:border-green-800' 
            : 'bg-red-50 border border-red-200 dark:bg-red-900/10 dark:border-red-800'
        }`}>
          <div className="flex items-center gap-2">
            {isHealthy ? (
              <CheckCircle className="w-4 h-4 text-green-600" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-red-600" />
            )}
            <span className={`text-xs font-medium ${
              isHealthy ? 'text-green-800 dark:text-green-400' : 'text-red-800 dark:text-red-400'
            }`}>
              {isHealthy 
                ? 'Training progressing normally' 
                : 'Weight updates not detected - check learning rate'}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}