export interface ModelStatus {
  id: number;
  name: string;
  status: 'idle' | 'training' | 'ready' | 'error';
  lastTrained?: string;
}

export interface TrainingProgress {
  step: number;
  loss: number;
  learningRate: number;
  gpuUtilization?: number;
  memoryUsage?: number;
  expertUtilization?: number[];
  layerSkipRate?: number;
  eta?: string;
}

export interface SystemMetrics {
  gpuUtilization: number;
  memoryUsage: number;
  memoryTotal: number;
  temperature: number;
  powerDraw: number;
}

export interface ExpertStats {
  utilization: number[];
  loadBalance: number;
  efficiency: number;
  skipRate: number;
}

export interface GenerationResult {
  id: number;
  output: string;
  tokensGenerated: number;
  generationTime: number;
  timestamp: string;
}

export interface CheckpointInfo {
  id: number;
  name: string;
  step: number;
  loss: number;
  size: number;
  isBest: boolean;
  createdAt: string;
  description?: string;
}

export interface TrainingMetric {
  step: number;
  loss: number;
  learningRate: number;
  timestamp: string;
}

export interface WebSocketMessage {
  type: 'training_progress' | 'training_complete' | 'training_error' | 'system_metrics';
  data: any;
}

export interface ParameterRange {
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface ParameterConfig {
  label: string;
  description?: string;
  type: 'number' | 'boolean' | 'select' | 'range';
  range?: ParameterRange;
  options?: { value: string | number; label: string }[];
}

export interface ModelPreset {
  name: string;
  description: string;
  config: Record<string, any>;
  size: string;
}
