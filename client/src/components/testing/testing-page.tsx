import { FunctionalityTester } from './functionality-tester';

export function TestingPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Functionality Testing
        </h2>
        <p className="text-muted-foreground">
          Test all interactive components and verify their functionality
        </p>
      </div>
      
      <FunctionalityTester />
    </div>
  );
}