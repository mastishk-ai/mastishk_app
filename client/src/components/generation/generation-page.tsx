import { useState } from "react";
import { TextGenerator } from "./text-generator";
import { GenerationSettings } from "./generation-settings";
import { GenerationHistory } from "./generation-history";
import { GenerationConfig } from "@shared/schema";

export function GenerationPage() {
  const [generationConfig, setGenerationConfig] = useState<GenerationConfig>({
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
    max_length: 500,
    repetition_penalty: 1.1,
    length_penalty: 1.0,
    no_repeat_ngram_size: 3,
    do_sample: true,
    early_stopping: false,
    num_beams: 1,
    use_multi_token_prediction: false
  });

  const updateConfig = (updates: Partial<GenerationConfig>) => {
    setGenerationConfig(prev => ({ ...prev, ...updates }));
  };

  return (
    <div className="space-y-6">
      {/* Text Generator */}
      <TextGenerator />
      
      {/* Generation Settings */}
      <GenerationSettings config={generationConfig} onUpdate={updateConfig} />
      
      {/* Generation History */}
      <GenerationHistory />
    </div>
  );
}
