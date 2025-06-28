import { DataUpload } from "./data-upload";
import { TrainingConfig } from "./training-config";
import { TrainingControls } from "./training-controls";

export function TrainingPage() {
  return (
    <div className="space-y-6">
      {/* Training Data Upload */}
      <DataUpload />
      
      {/* Training Configuration */}
      <TrainingConfig />
      
      {/* Training Controls */}
      <TrainingControls />
    </div>
  );
}
