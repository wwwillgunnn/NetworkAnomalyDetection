import { Button } from "@/components/ui/button";

export default function HomePage() {
  return (
    <main className="min-h-screen flex flex-col bg-gray-50">
      {/* Hero */}
      <section className="flex flex-col items-center justify-center text-center py-20 bg-gradient-to-r from-indigo-600 to-blue-500 text-white">
        <h1 className="text-5xl font-bold mb-4">
          Intelligent Anomaly Detection
        </h1>
        <p className="text-lg max-w-xl mb-6">
          Monitor and detect suspicious activity in your network in real-time
          using advanced machine learning techniques.
        </p>
        <Button variant="secondary" size="lg" asChild>
          <a href="/dashboard">Go to Dashboard</a>
        </Button>
      </section>

      {/* Features */}
      <section className="py-16 px-6 md:px-20 grid md:grid-cols-3 gap-8 text-center">
        <div>
          <h3 className="text-xl font-semibold">🚀 Real-Time Monitoring</h3>
          <p className="text-gray-600 mt-2">
            Detect anomalies in your network traffic as they happen.
          </p>
        </div>
        <div>
          <h3 className="text-xl font-semibold">🔍 ML-Powered Detection</h3>
          <p className="text-gray-600 mt-2">
            Using Isolation Forests, Autoencoders, and advanced algorithms.
          </p>
        </div>
        <div>
          <h3 className="text-xl font-semibold">📊 Visual Dashboard</h3>
          <p className="text-gray-600 mt-2">
            Intuitive dashboard with charts, tables, and alerting features.
          </p>
        </div>
      </section>

      {/* Preview */}
      <section className="py-16 bg-white flex flex-col items-center">
        <h2 className="text-3xl font-bold mb-6">See it in Action</h2>
        <div className="border rounded-lg shadow-lg w-[80%] h-96 flex items-center justify-center text-gray-400">
          [ Dashboard Preview Screenshot ]
        </div>
      </section>

      {/* Footer */}
      <footer className="py-6 text-center text-gray-500 text-sm">
        © 2025 Intelligent Anomaly Detection Project – Built with Next.js &
        FastAPI
      </footer>
    </main>
  );
}
