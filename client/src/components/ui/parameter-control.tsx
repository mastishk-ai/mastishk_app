import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ParameterConfig } from "@/lib/types";

interface ParameterControlProps {
  config: ParameterConfig;
  value: any;
  onChange: (value: any) => void;
  className?: string;
}

export function ParameterControl({ config, value, onChange, className }: ParameterControlProps) {
  const { label, description, type, range, options } = config;

  const renderControl = () => {
    switch (type) {
      case 'boolean':
        return (
          <div className="flex items-center space-x-2">
            <Switch
              checked={value}
              onCheckedChange={(checked) => {
                console.log(`Switch ${label} toggled to:`, checked);
                onChange(checked);
              }}
            />
            <Label className="text-sm">{label}</Label>
          </div>
        );

      case 'select':
        return (
          <div className="space-y-2">
            <Label className="text-sm font-medium">{label}</Label>
            <Select value={value?.toString()} onValueChange={(val) => onChange(isNaN(Number(val)) ? val : Number(val))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {options?.map((option) => (
                  <SelectItem key={option.value} value={option.value.toString()}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {description && <p className="text-xs text-muted-foreground">{description}</p>}
          </div>
        );

      case 'range':
        if (!range) return null;
        return (
          <div className="space-y-2">
            <Label className="text-sm font-medium">{label}</Label>
            <div className="flex items-center space-x-3">
              <Slider
                value={[value]}
                onValueChange={(values) => {
                  console.log(`🎚️ Slider ${label} changed to:`, values[0]);
                  console.log('🎚️ Calling onChange with:', values[0]);
                  onChange(values[0]);
                }}
                min={range.min}
                max={range.max}
                step={range.step}
                className="flex-1"
              />
              <Input
                type="number"
                value={value}
                onChange={(e) => {
                  console.log(`Input ${label} changed to:`, e.target.value);
                  onChange(Number(e.target.value));
                }}
                min={range.min}
                max={range.max}
                step={range.step}
                className="w-20"
              />
            </div>
            {description && <p className="text-xs text-muted-foreground">{description}</p>}
          </div>
        );

      case 'number':
      default:
        return (
          <div className="space-y-2">
            <Label className="text-sm font-medium">{label}</Label>
            <Input
              type="number"
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
              step={range?.step || 'any'}
            />
            {description && <p className="text-xs text-muted-foreground">{description}</p>}
          </div>
        );
    }
  };

  return (
    <div className={`parameter-control ${className}`}>
      {renderControl()}
    </div>
  );
}
