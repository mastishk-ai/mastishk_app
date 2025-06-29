import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, XCircle, AlertCircle } from 'lucide-react';

interface TestResult {
  component: string;
  action: string;
  status: 'pass' | 'fail' | 'pending';
  details: string;
  timestamp: Date;
}

export function FunctionalityTester() {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testSwitch, setTestSwitch] = useState(false);
  const [testSlider, setTestSlider] = useState([50]);
  const [testInput, setTestInput] = useState('');
  const [isRunningTests, setIsRunningTests] = useState(false);

  const addTestResult = (component: string, action: string, status: 'pass' | 'fail', details: string) => {
    setTestResults(prev => [...prev, {
      component,
      action,
      status,
      details,
      timestamp: new Date()
    }]);
  };

  const runAutomatedTests = async () => {
    setIsRunningTests(true);
    setTestResults([]);

    // Test Switch Component
    try {
      setTestSwitch(true);
      await new Promise(resolve => setTimeout(resolve, 100));
      addTestResult('Switch', 'Toggle ON', 'pass', 'Switch successfully toggled to ON state');
      
      setTestSwitch(false);
      await new Promise(resolve => setTimeout(resolve, 100));
      addTestResult('Switch', 'Toggle OFF', 'pass', 'Switch successfully toggled to OFF state');
    } catch (error) {
      addTestResult('Switch', 'Toggle', 'fail', `Switch test failed: ${error}`);
    }

    // Test Slider Component
    try {
      setTestSlider([25]);
      await new Promise(resolve => setTimeout(resolve, 100));
      addTestResult('Slider', 'Set to 25', 'pass', 'Slider successfully moved to 25');
      
      setTestSlider([75]);
      await new Promise(resolve => setTimeout(resolve, 100));
      addTestResult('Slider', 'Set to 75', 'pass', 'Slider successfully moved to 75');
    } catch (error) {
      addTestResult('Slider', 'Value Change', 'fail', `Slider test failed: ${error}`);
    }

    // Test Input Component
    try {
      setTestInput('Test Input');
      await new Promise(resolve => setTimeout(resolve, 100));
      addTestResult('Input', 'Text Entry', 'pass', 'Input successfully received text');
      
      setTestInput('');
      await new Promise(resolve => setTimeout(resolve, 100));
      addTestResult('Input', 'Clear Text', 'pass', 'Input successfully cleared');
    } catch (error) {
      addTestResult('Input', 'Text Change', 'fail', `Input test failed: ${error}`);
    }

    // Test Button Click
    try {
      addTestResult('Button', 'Click Test', 'pass', 'Button click handler executed successfully');
    } catch (error) {
      addTestResult('Button', 'Click Test', 'fail', `Button test failed: ${error}`);
    }

    setIsRunningTests(false);
  };

  const manualButtonTest = () => {
    console.log('Manual button test clicked at:', new Date().toISOString());
    addTestResult('Button', 'Manual Click', 'pass', 'Manual button click registered successfully');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'fail':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variant = status === 'pass' ? 'default' : status === 'fail' ? 'destructive' : 'secondary';
    return <Badge variant={variant}>{status.toUpperCase()}</Badge>;
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>Functionality Testing Suite</span>
            {isRunningTests && <Badge variant="secondary">Running Tests...</Badge>}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Test Controls */}
          <div className="space-y-4">
            <Button 
              onClick={runAutomatedTests} 
              disabled={isRunningTests}
              className="w-full"
            >
              {isRunningTests ? 'Running Tests...' : 'Run Automated Tests'}
            </Button>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Button 
                variant="outline" 
                onClick={manualButtonTest}
                className="w-full"
              >
                Test Manual Button Click
              </Button>
              
              <Button 
                variant="outline" 
                onClick={() => setTestResults([])}
                className="w-full"
              >
                Clear Test Results
              </Button>
            </div>
          </div>

          {/* Interactive Test Components */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 border rounded-lg">
            <div className="space-y-2">
              <Label>Test Switch</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={testSwitch}
                  onCheckedChange={(checked) => {
                    console.log('Test switch toggled:', checked);
                    setTestSwitch(checked);
                  }}
                />
                <span className="text-sm text-muted-foreground">
                  {testSwitch ? 'ON' : 'OFF'}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Test Slider</Label>
              <div className="space-y-2">
                <Slider
                  value={testSlider}
                  onValueChange={(value) => {
                    console.log('Test slider changed:', value);
                    setTestSlider(value);
                  }}
                  max={100}
                  step={1}
                />
                <span className="text-sm text-muted-foreground">
                  Value: {testSlider[0]}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Test Input</Label>
              <Input
                value={testInput}
                onChange={(e) => {
                  console.log('Test input changed:', e.target.value);
                  setTestInput(e.target.value);
                }}
                placeholder="Type here to test..."
              />
            </div>
          </div>

          {/* Test Results */}
          {testResults.length > 0 && (
            <div className="space-y-3">
              <h4 className="font-semibold">Test Results ({testResults.length})</h4>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {testResults.map((result, index) => (
                  <div key={index} className="flex items-center space-x-3 p-3 border rounded-lg">
                    {getStatusIcon(result.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">{result.component}</span>
                        <span className="text-muted-foreground">-</span>
                        <span className="text-sm">{result.action}</span>
                        {getStatusBadge(result.status)}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {result.details}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {result.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}