import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";
import Dashboard from "@/pages/dashboard";
import VisualizationPage from "@/pages/visualization";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Dashboard} />
      <Route path="/training" component={Dashboard} />
      <Route path="/generation" component={Dashboard} />
      <Route path="/monitoring" component={Dashboard} />
      <Route path="/checkpoints" component={Dashboard} />
      <Route path="/analytics" component={Dashboard} />
      <Route path="/visualization" component={VisualizationPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div className="dark">
          <Toaster />
          <Router />
        </div>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
