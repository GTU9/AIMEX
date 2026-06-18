import { apiClient } from '../api';

export interface InfluencerDocument {
  documents_id: string;
  documents_name: string;
  file_size?: number;
  file_path: string;
  is_vectorized: number;
  created_at?: string;
}

export interface DocumentListResponse {
  documents: InfluencerDocument[];
  total_count: number;
}

export interface DocumentUploadResponse {
  success: boolean;
  message: string;
  documents_id?: string;
  file_path?: string;
  file_size?: number;
}

/**
 * 인플루언서별 RAG 문서 관리 (신규 인플루언서 스코프 엔드포인트)
 * - 업로드(로컬 저장) → 벡터화(Modal 임베딩 + Milvus) → 챗봇 RAG 활용
 */
export class DocumentService {
  /** 인플루언서의 문서 목록 */
  static async list(influencerId: string): Promise<DocumentListResponse> {
    return await apiClient.get<DocumentListResponse>(
      `/api/v1/documents/by-influencer/${influencerId}`
    );
  }

  /** 문서 업로드 (PDF/TXT/MD) */
  static async upload(file: File, influencerId: string): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('influencer_id', influencerId);
    return await apiClient.post<DocumentUploadResponse>(
      '/api/v1/documents/upload',
      formData,
      { headers: {} } // FormData는 Content-Type 자동 설정
    );
  }

  /** 문서 벡터화 (청킹 → 임베딩 → Milvus 저장) */
  static async vectorize(
    documentsId: string
  ): Promise<{ documents_id: string; chunks: number; is_vectorized: number }> {
    return await apiClient.post(`/api/v1/documents/${documentsId}/vectorize`, {});
  }

  /** 문서 삭제 */
  static async remove(documentsId: string): Promise<unknown> {
    return await apiClient.delete(`/api/v1/documents/${documentsId}`);
  }
}
