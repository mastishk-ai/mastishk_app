import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Settings, 
  Play, 
  Edit3, 
  Activity, 
  Save, 
  BarChart2, 
  CheckCircle2,
  Code,
  Zap,
  Globe,
  Monitor,
  Cpu,
  Network,
  Server,
  BookOpen
} from 'lucide-react';

export function DocumentationPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground mb-2">
          Mastishk© Transformer Studio Documentation
        </h1>
        <p className="text-muted-foreground">
          Complete guide to using the advanced transformer experimentation platform
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Table of Contents */}
        <div className="lg:col-span-1">
          <Card className="sticky top-6">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <BookOpen className="w-5 h-5" />
                Quick Guide
              </CardTitle>
            </CardHeader>
            <CardContent>
              <nav className="space-y-2 text-sm">
                <a href="#overview" className="block text-primary hover:underline cursor-pointer">Overview</a>
                <a href="#getting-started" className="block text-primary hover:underline cursor-pointer">Getting Started</a>
                <a href="#model-config" className="block text-primary hover:underline cursor-pointer">Model Configuration</a>
                <a href="#training" className="block text-primary hover:underline cursor-pointer">Training Pipeline</a>
                <a href="#generation" className="block text-primary hover:underline cursor-pointer">Text Generation</a>
                <a href="#testing" className="block text-primary hover:underline cursor-pointer">Testing Suite</a>
                <a href="#troubleshooting" className="block text-primary hover:underline cursor-pointer">Troubleshooting</a>
              </nav>
            </CardContent>
          </Card>
        </div>

        {/* Main Documentation Content */}
        <div className="lg:col-span-3">
          <ScrollArea className="h-[80vh] pr-4">
            <div className="space-y-8">
              
              {/* Overview */}
              <section id="overview">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Globe className="w-5 h-5" />
                      Overview
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      Mastishk© Transformer Studio is an advanced transformer experimentation platform that provides 
                      a comprehensive environment for training, configuring, and experimenting with sophisticated 
                      transformer models. Built with React and Node.js, it features real-time monitoring, 
                      3D visualizations, and advanced ML capabilities.
                    </p>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-3 border rounded-lg">
                        <Settings className="w-8 h-8 mx-auto mb-2 text-primary" />
                        <div className="font-semibold">Model Config</div>
                        <div className="text-xs text-muted-foreground">Advanced architecture setup</div>
                      </div>
                      <div className="text-center p-3 border rounded-lg">
                        <Play className="w-8 h-8 mx-auto mb-2 text-primary" />
                        <div className="font-semibold">Training</div>
                        <div className="text-xs text-muted-foreground">Real-time pipeline</div>
                      </div>
                      <div className="text-center p-3 border rounded-lg">
                        <Edit3 className="w-8 h-8 mx-auto mb-2 text-primary" />
                        <div className="font-semibold">Generation</div>
                        <div className="text-xs text-muted-foreground">Text synthesis</div>
                      </div>
                      <div className="text-center p-3 border rounded-lg">
                        <BarChart2 className="w-8 h-8 mx-auto mb-2 text-primary" />
                        <div className="font-semibold">Analytics</div>
                        <div className="text-xs text-muted-foreground">Performance insights</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                      <div>
                        <h4 className="font-semibold mb-2">Key Features</h4>
                        <ul className="text-sm space-y-1 text-muted-foreground">
                          <li>• Advanced transformer architectures</li>
                          <li>• MoE (Mixture of Experts) support</li>
                          <li>• Flash Attention integration</li>
                          <li>• Real-time training monitoring</li>
                          <li>• 3D model visualization</li>
                          <li>• Checkpoint management</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">Supported Models</h4>
                        <ul className="text-sm space-y-1 text-muted-foreground">
                          <li>• Standard Transformer</li>
                          <li>• Generative Transformer</li>
                          <li>• Mixture of Experts (MoE)</li>
                          <li>• Mixture of Depths (MoD)</li>
                          <li>• Multi-token prediction</li>
                          <li>• Custom architectures</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">Advanced Features</h4>
                        <ul className="text-sm space-y-1 text-muted-foreground">
                          <li>• Differential Attention</li>
                          <li>• MiniMax Optimization</li>
                          <li>• LoLCATs Compression</li>
                          <li>• Weight verification</li>
                          <li>• Gradient checkpointing</li>
                          <li>• Mixed precision training</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Getting Started */}
              <section id="getting-started">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="w-5 h-5" />
                      Getting Started
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-3">
                      <div className="flex items-start gap-3">
                        <Badge variant="outline" className="mt-0.5">1</Badge>
                        <div>
                          <div className="font-medium">Configure Model</div>
                          <div className="text-sm text-muted-foreground">
                            Set up your transformer architecture, choose model size, and enable advanced features
                          </div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Badge variant="outline" className="mt-0.5">2</Badge>
                        <div>
                          <div className="font-medium">Upload Training Data</div>
                          <div className="text-sm text-muted-foreground">
                            Upload your training data in TXT, JSON, JSONL, or CSV format
                          </div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Badge variant="outline" className="mt-0.5">3</Badge>
                        <div>
                          <div className="font-medium">Start Training</div>
                          <div className="text-sm text-muted-foreground">
                            Configure training parameters and begin the training process
                          </div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Badge variant="outline" className="mt-0.5">4</Badge>
                        <div>
                          <div className="font-medium">Monitor Progress</div>
                          <div className="text-sm text-muted-foreground">
                            Watch real-time training metrics and performance graphs
                          </div>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Badge variant="outline" className="mt-0.5">5</Badge>
                        <div>
                          <div className="font-medium">Generate Text</div>
                          <div className="text-sm text-muted-foreground">
                            Use your trained model to generate text with various strategies
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Model Configuration */}
              <section id="model-config">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Settings className="w-5 h-5" />
                      Model Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      The Model Configuration page is where you define your transformer architecture. 
                      It provides comprehensive options for customizing every aspect of your model.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold mb-3">Basic Parameters</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Model Size:</span>
                            <span>Choose from Tiny to 13B parameters</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Architecture:</span>
                            <span>Standard, Generative, MoE, MoD</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Vocabulary Size:</span>
                            <span>32K - 128K tokens</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Context Length:</span>
                            <span>512 - 8192 tokens</span>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold mb-3">Advanced Features</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                            <span>Flash Attention - Memory-efficient attention</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                            <span>Differential Attention - Enhanced mechanism</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                            <span>MiniMax Optimization - Adversarial training</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                            <span>LoLCATs Compression - Low-rank compression</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                            <span>Multi-Token Prediction - Predict multiple tokens</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                      <div className="font-medium text-blue-900 dark:text-blue-100 mb-2">Configuration Tips</div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <div className="font-medium text-blue-900 dark:text-blue-100">For Beginners</div>
                          <div className="text-sm text-blue-700 dark:text-blue-300">
                            Start with "Tiny" model size and standard architecture. Enable Flash Attention for better performance.
                          </div>
                        </div>
                        <div>
                          <div className="font-medium text-blue-900 dark:text-blue-100">For Large Datasets</div>
                          <div className="text-sm text-blue-700 dark:text-blue-300">
                            Use "Large" or "XL" models with MoE architecture. Enable gradient checkpointing to save memory.
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Training Pipeline */}
              <section id="training">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Play className="w-5 h-5" />
                      Training Pipeline
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      The Training Pipeline manages data upload, training configuration, and model training execution 
                      with real-time monitoring and automatic checkpoint saving.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 border rounded-lg">
                        <h5 className="font-semibold mb-2">Data Upload</h5>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          <li>• Supports TXT, JSON, JSONL, CSV</li>
                          <li>• Automatic format detection</li>
                          <li>• Data validation and preprocessing</li>
                          <li>• Progress tracking</li>
                        </ul>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <h5 className="font-semibold mb-2">Training Config</h5>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          <li>• Learning rate scheduling</li>
                          <li>• Batch size optimization</li>
                          <li>• Optimizer selection</li>
                          <li>• Mixed precision training</li>
                        </ul>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <h5 className="font-semibold mb-2">Advanced Options</h5>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          <li>• Gradient checkpointing</li>
                          <li>• Early stopping</li>
                          <li>• Weight verification</li>
                          <li>• Checkpoint integrity</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Text Generation */}
              <section id="generation">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Edit3 className="w-5 h-5" />
                      Text Generation
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      Generate text using your trained models with various sampling strategies and generation parameters.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold mb-3">Generation Strategies</h4>
                        <div className="space-y-3">
                          <div className="p-3 border rounded-lg">
                            <div className="font-medium">Greedy Decoding</div>
                            <div className="text-sm text-muted-foreground">
                              Always selects the most probable token. Fast but can be repetitive.
                            </div>
                          </div>
                          <div className="p-3 border rounded-lg">
                            <div className="font-medium">Beam Search</div>
                            <div className="text-sm text-muted-foreground">
                              Explores multiple paths simultaneously. Good for coherent text.
                            </div>
                          </div>
                          <div className="p-3 border rounded-lg">
                            <div className="font-medium">Nucleus Sampling</div>
                            <div className="text-sm text-muted-foreground">
                              Samples from top-p probability mass. Balances quality and diversity.
                            </div>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold mb-3">Parameters</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Temperature:</span>
                            <span>0.1 - 2.0 (creativity control)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Top-p:</span>
                            <span>0.1 - 1.0 (nucleus sampling)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Top-k:</span>
                            <span>1 - 100 (candidate tokens)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Max Length:</span>
                            <span>1 - 2048 tokens</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Repetition Penalty:</span>
                            <span>1.0 - 2.0 (avoid repetition)</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Testing Suite */}
              <section id="testing">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5" />
                      Testing Suite
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      The Testing Suite provides comprehensive tools to verify component functionality 
                      and ensure all interactive elements are working correctly.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-semibold mb-3">Testing Methods</h4>
                        <div className="space-y-3">
                          <div className="p-3 border rounded-lg">
                            <div className="font-medium">Console Logging</div>
                            <div className="text-sm text-muted-foreground">
                              Real-time logs in browser console (F12) for all component interactions
                            </div>
                          </div>
                          <div className="p-3 border rounded-lg">
                            <div className="font-medium">Automated Tests</div>
                            <div className="text-sm text-muted-foreground">
                              Run comprehensive test suite to verify all functionality at once
                            </div>
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold mb-3">What Gets Tested</h4>
                        <ul className="text-sm text-muted-foreground space-y-2">
                          <li>• Toggle switches (on/off states)</li>
                          <li>• Sliders (value changes)</li>
                          <li>• Buttons (click handlers)</li>
                          <li>• Input fields (text entry)</li>
                          <li>• Form submissions</li>
                          <li>• Navigation links</li>
                          <li>• Theme switching</li>
                          <li>• File uploads</li>
                          <li>• Real-time updates</li>
                        </ul>
                      </div>
                    </div>

                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                      <div className="font-medium text-green-900 dark:text-green-100 mb-2">Testing Workflow</div>
                      <ol className="text-sm text-green-700 dark:text-green-300 space-y-1">
                        <li>1. Open Testing Suite from sidebar</li>
                        <li>2. Run automated tests to verify basic functionality</li>
                        <li>3. Use manual test components for specific checks</li>
                        <li>4. Monitor console logs for real-time feedback</li>
                        <li>5. Check test results for pass/fail status</li>
                      </ol>
                    </div>
                  </CardContent>
                </Card>
              </section>

              {/* Troubleshooting */}
              <section id="troubleshooting">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      Troubleshooting
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>
                      Common issues and their solutions for Mastishk© Transformer Studio.
                    </p>

                    <div className="space-y-6">
                      <div className="p-4 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-950">
                        <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">UI Components Not Responding</h4>
                        <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                          <li>• Open browser console (F12) to check for errors</li>
                          <li>• Refresh the page to reset state</li>
                          <li>• Use the Testing Suite to verify functionality</li>
                          <li>• Check if JavaScript is enabled</li>
                        </ul>
                      </div>

                      <div className="p-4 border-l-4 border-green-500 bg-green-50 dark:bg-green-950">
                        <h4 className="font-semibold text-green-900 dark:text-green-100 mb-2">WebSocket Connection Issues</h4>
                        <ul className="text-sm text-green-700 dark:text-green-300 space-y-1">
                          <li>• Check internet connection stability</li>
                          <li>• Verify firewall settings allow WebSocket</li>
                          <li>• Try refreshing the page</li>
                          <li>• Check if real-time updates are working</li>
                        </ul>
                      </div>
                    </div>

                    <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <h4 className="font-semibold mb-2">Getting Help</h4>
                      <p className="text-sm text-muted-foreground">
                        If you continue to experience issues, check the console logs (F12) for detailed error messages. 
                        The Testing Suite can help identify specific component problems. For persistent issues, 
                        try refreshing the page or restarting your browser.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </section>

            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}