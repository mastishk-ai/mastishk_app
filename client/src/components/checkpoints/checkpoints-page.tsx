import { CheckpointList } from "./checkpoint-list";
import { StorageManagement } from "./storage-management";

export function CheckpointsPage() {
  return (
    <div className="space-y-6">
      {/* Checkpoint List */}
      <CheckpointList />
      
      {/* Storage Management */}
      <StorageManagement />
    </div>
  );
}
