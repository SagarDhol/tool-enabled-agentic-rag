import ChatInterface from '@/components/ChatInterface';
import DocumentUpload from '@/components/DocumentUpload';
import { CubeIcon } from '@heroicons/react/24/outline';

export default function Home() {
  return (
    <main className="flex h-screen bg-background overflow-hidden">
      {/* Sidebar / Document Management */}
      <aside className="w-80 border-r border-gray-800 flex flex-col p-6 space-y-8 bg-[#0d0f11]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <CubeIcon className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-bold font-outfit tracking-tight">Agentic RAG</h1>
        </div>

        <section className="space-y-4">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-widest pl-1">Knowledge Base</h2>
          <DocumentUpload />
        </section>

        <section className="mt-auto pt-6 border-t border-gray-800">
          <div className="flex items-center gap-3 p-3 rounded-xl hover:bg-gray-800 transition-colors cursor-pointer group">
            <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-sm font-medium">JD</div>
            <div className="flex-1">
              <p className="text-sm font-medium">Guest User</p>
              <p className="text-xs text-gray-500">Free Tier</p>
            </div>
          </div>
        </section>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative">
        <header className="h-16 border-b border-gray-800 flex items-center px-8 justify-between glass-morphism z-10">
          <div className="flex items-center gap-4">
            <span className="flex h-2 w-2 rounded-full bg-green-500"></span>
            <span className="text-sm font-medium text-gray-300">Agent Online</span>
          </div>
          <div className="text-xs text-gray-500 font-mono">Ollama (Llama 3) Local Agent</div>
        </header>

        <div className="flex-1 relative">
          <ChatInterface />
        </div>
      </div>
    </main>
  );
}
