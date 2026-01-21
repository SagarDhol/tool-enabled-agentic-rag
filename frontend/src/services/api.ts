export interface Source {
  type: 'document' | 'tool';
  name: string;
  reference: string;
}

export interface AgentResponse {
  answer: str;
  sources: Source[];
  confidence: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = {
  async chat(query: str, threadId?: str): Promise<AgentResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, thread_id: threadId }),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch chat response');
    }

    return response.json();
  },

  async upload(file: File): Promise<{ message: str; chunks: number } | { error: str }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to upload document');
    }

    return response.json();
  },

  async checkHealth(): Promise<{ status: str }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  }
};
