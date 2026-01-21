'use client';

import React, { useState } from 'react';
import { api } from '@/services/api';
import { ArrowUpTrayIcon, DocumentIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';

export default function DocumentUpload() {
    const [file, setFile] = useState<File | null>(null);
    const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
    const [message, setMessage] = useState('');

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setStatus('idle');
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setStatus('uploading');
        try {
            const result = await api.upload(file);
            if ('error' in result) {
                setStatus('error');
                setMessage(result.error);
            } else {
                setStatus('success');
                setMessage(result.message);
                setFile(null);
            }
        } catch (error) {
            setStatus('error');
            setMessage('Failed to upload document.');
        }
    };

    return (
        <div className="glass-morphism p-6 rounded-2xl border border-gray-800 google-shadow max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 font-outfit">
                <ArrowUpTrayIcon className="w-5 h-5 text-blue-500" />
                Ingest Knowledge
            </h3>

            <div className="space-y-4">
                <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-700 rounded-xl cursor-pointer hover:border-blue-500 hover:bg-gray-800/50 transition-all duration-200">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        {file ? (
                            <>
                                <DocumentIcon className="w-8 h-8 text-blue-500 mb-2" />
                                <p className="text-sm text-gray-300 font-medium">{file.name}</p>
                                <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            </>
                        ) : (
                            <>
                                <ArrowUpTrayIcon className="w-8 h-8 text-gray-500 mb-2" />
                                <p className="text-sm text-gray-400">Click to upload PDF or TXT</p>
                            </>
                        )}
                    </div>
                    <input type="file" className="hidden" onChange={handleFileChange} accept=".pdf,.txt" />
                </label>

                <button
                    onClick={handleUpload}
                    disabled={!file || status === 'uploading'}
                    className={`w-full py-3 rounded-xl font-medium transition-all duration-200 flex items-center justify-center gap-2 ${!file || status === 'uploading'
                            ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                >
                    {status === 'uploading' ? (
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    ) : (
                        'Process Document'
                    )}
                </button>

                {status === 'success' && (
                    <div className="p-3 bg-green-900/20 border border-green-800 rounded-lg flex items-center gap-2 text-green-400 text-sm">
                        <CheckCircleIcon className="w-5 h-5" />
                        {message}
                    </div>
                )}

                {status === 'error' && (
                    <div className="p-3 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-2 text-red-400 text-sm">
                        <XCircleIcon className="w-5 h-5" />
                        {message}
                    </div>
                )}
            </div>
        </div>
    );
}
