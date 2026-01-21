'use client';

import React, { useState, useRef, useEffect } from 'react';
import { api, AgentResponse, Source } from '@/services/api';
import { PaperAirplaneIcon, DocumentTextIcon, WrenchScrewdriverIcon, CubeIcon } from '@heroicons/react/24/outline';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
    id: string;
    role: 'user' | 'agent';
    content: string;
    sources?: Source[];
    confidence?: number;
    timestamp: Date;
}

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await api.chat(input);

            // Sanitization: If backend returns raw JSON string as 'answer'
            let cleanAnswer = response.answer;
            if (typeof cleanAnswer === 'string' && cleanAnswer.trim().startsWith('{')) {
                try {
                    const parsed = JSON.parse(cleanAnswer);
                    if (parsed.answer) cleanAnswer = parsed.answer;
                } catch (e) {
                    // Not valid JSON or parsing failed, stick with original
                }
            }

            const agentMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'agent',
                content: cleanAnswer,
                sources: response.sources,
                confidence: response.confidence,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, agentMessage]);
        } catch (error) {
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'agent',
                content: 'Sorry, I encountered an error while processing your request.',
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth"
            >
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                        <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center google-shadow">
                            <CubeIcon className="w-10 h-10 text-white" />
                        </div>
                        <h2 className="text-2xl font-bold font-outfit">How can I help you today?</h2>
                        <p className="text-gray-400 max-w-md">
                            Ask me anything! I can retrieve information from your documents,
                            perform calculations, and reason through complex questions.
                        </p>
                    </div>
                )}
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
                    >
                        <div className={`max-w-[85%] p-4 ${message.role === 'user'
                            ? 'chat-bubble-user text-white'
                            : 'chat-bubble-agent text-gray-100'
                            }`}>
                            <div className="prose prose-invert max-w-none leading-relaxed text-sm lg:text-base">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {message.content}
                                </ReactMarkdown>
                            </div>

                            {message.sources && message.sources.length > 0 && (
                                <div className="mt-4 pt-4 border-t border-gray-800">
                                    <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Sources</p>
                                    <div className="flex flex-wrap">
                                        {message.sources.map((source, idx) => (
                                            <span key={idx} className="source-tag">
                                                {source.type === 'document' ? (
                                                    <DocumentTextIcon className="w-3 h-3 mr-1" />
                                                ) : (
                                                    <WrenchScrewdriverIcon className="w-3 h-3 mr-1" />
                                                )}
                                                {source.name} {source.reference ? `(${source.reference})` : ''}
                                            </span>
                                        ))}
                                    </div>
                                    {message.confidence && (
                                        <div className="mt-2 text-[10px] text-gray-500">
                                            Confidence Score: {(message.confidence * 100).toFixed(1)}%
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="chat-bubble-agent p-4 space-y-2 opacity-50">
                            <div className="h-2 w-24 bg-gray-700 rounded animate-pulse"></div>
                            <div className="h-2 w-48 bg-gray-700 rounded animate-pulse"></div>
                        </div>
                    </div>
                )}
            </div>

            <div className="p-4 border-t border-gray-800 glass-morphism sticky bottom-0">
                <form onSubmit={handleSubmit} className="relative flex items-center">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question..."
                        className="w-full bg-input-bg border-none rounded-2xl py-4 pl-6 pr-14 text-gray-100 focus:ring-2 focus:ring-blue-600 focus:outline-none transition-all duration-200"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 p-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:bg-gray-700 transition-all duration-200"
                    >
                        <PaperAirplaneIcon className="w-5 h-5" />
                    </button>
                </form>
                <p className="text-[10px] text-center text-gray-500 mt-2">
                    Agentic RAG System may provide inaccurate info. Verify important facts.
                </p>
            </div>
        </div>
    );
}
