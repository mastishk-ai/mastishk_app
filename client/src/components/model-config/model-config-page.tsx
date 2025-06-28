import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { ArchitectureConfig } from "./architecture-config";
import { MoeConfig } from "./moe-config";
import { ModConfig } from "./mod-config";
import { ModelPresets } from "./model-presets";
import { ModelConfig } from "@shared/schema";

interface ModelConfigPageProps {
  config: ModelConfig;
  onUpdate: (updates: Partial<ModelConfig>) => void;
  onUpdateMoe: (updates: Partial<NonNullable<ModelConfig['moe_config']>>) => void;
  onUpdateMod: (updates: Partial<NonNullable<ModelConfig['mod_config']>>) => void;
  onCreateModel: (data: { name: string }) => void;
  isCreating: boolean;
  validateConfig: () => string[];
}

export function ModelConfigPage({
  config,
  onUpdate,
  onUpdateMoe,
  onUpdateMod,
  onCreateModel,
  isCreating,
  validateConfig
}: ModelConfigPageProps) {
  const [modelName, setModelName] = useState("");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const { toast } = useToast();

  const handleCreateModel = () => {
    if (!modelName.trim()) {
      toast({
        title: "Validation Error",
        description: "Please enter a model name",
        variant: "destructive"
      });
      return;
    }

    const errors = validateConfig();
    if (errors.length > 0) {
      toast({
        title: "Configuration Error",
        description: errors[0],
        variant: "destructive"
      });
      return;
    }

    onCreateModel({ name: modelName });
    setShowCreateDialog(false);
    setModelName("");
  };

  const handleExportConfig = () => {
    const configBlob = new Blob([JSON.stringify(config, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(configBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mastishk_config_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Config Exported",
      description: "Configuration saved to file"
    });
  };

  const handleImportConfig = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedConfig = JSON.parse(e.target?.result as string);
        onUpdate(importedConfig);
        toast({
          title: "Config Imported",
          description: "Configuration loaded successfully"
        });
      } catch (error) {
        toast({
          title: "Import Error",
          description: "Invalid configuration file",
          variant: "destructive"
        });
      }
    };
    reader.readAsText(file);
  };

  const handleResetConfig = () => {
    onUpdate({
      vocab_size: 32000,
      hidden_size: 4096,
      intermediate_size: 11008,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      hidden_act: "silu",
      max_position_embeddings: 4096,
      initializer_range: 0.02,
      rms_norm_eps: 1e-5,
      use_flash_attention: true,
      use_differential_attention: false,
      use_minimax: false,
      lolcats_enabled: false,
      use_multi_token_prediction: false,
      use_moe: false,
      use_mod: false,
    });

    toast({
      title: "Config Reset",
      description: "Configuration reset to defaults"
    });
  };

  return (
    <div className="space-y-6">
      {/* Architecture Configuration */}
      <ArchitectureConfig config={config} onUpdate={onUpdate} />

      {/* MoE Configuration */}
      <MoeConfig config={config} onUpdate={onUpdate} onUpdateMoe={onUpdateMoe} />

      {/* MoD Configuration */}
      <ModConfig config={config} onUpdate={onUpdate} onUpdateMod={onUpdateMod} />

      {/* Model Presets and Actions */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Quick Presets */}
        <ModelPresets onApplyPreset={onUpdate} />
        
        {/* Configuration Actions */}
        <div className="flex-1 bg-card rounded-xl border p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
            <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-3a2 2 0 00-2 2v1a2 2 0 01-2 2H9a2 2 0 01-2-2v-1a2 2 0 00-2-2H2" />
            </svg>
            Configuration
          </h3>
          <div className="space-y-3">
            <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
              <DialogTrigger asChild>
                <Button className="w-full bg-blue-600 hover:bg-blue-700">
                  Create Model
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Model</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="model-name">Model Name</Label>
                    <Input
                      id="model-name"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      placeholder="Enter model name..."
                    />
                  </div>
                  <div className="flex justify-end space-x-2">
                    <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                      Cancel
                    </Button>
                    <Button 
                      onClick={handleCreateModel}
                      disabled={isCreating}
                    >
                      {isCreating ? 'Creating...' : 'Create Model'}
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>

            <div className="flex space-x-2">
              <Button 
                variant="outline" 
                className="flex-1"
                onClick={handleExportConfig}
              >
                Export JSON
              </Button>
              <Button 
                variant="outline" 
                className="flex-1"
                onClick={() => document.getElementById('config-import')?.click()}
              >
                Import JSON
              </Button>
              <input
                id="config-import"
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImportConfig}
              />
            </div>
            
            <Button 
              variant="outline" 
              className="w-full"
              onClick={handleResetConfig}
            >
              Reset to Defaults
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
