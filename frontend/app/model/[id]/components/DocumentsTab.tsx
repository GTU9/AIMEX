"use client";

import { useEffect, useRef, useState } from "react";
import { BookOpen, Upload, Loader2, Trash2, CheckCircle2, Circle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DocumentService, type InfluencerDocument } from "@/lib/services/document.service";

interface DocumentsTabProps {
  influencerId: string;
}

export default function DocumentsTab({ influencerId }: DocumentsTabProps) {
  const [docs, setDocs] = useState<InfluencerDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [msg, setMsg] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);

  const load = async () => {
    try {
      setLoading(true);
      const res = await DocumentService.list(influencerId);
      setDocs(res.documents || []);
    } catch (e) {
      console.error(e);
      setDocs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (influencerId) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [influencerId]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setMsg("");
    try {
      const up = await DocumentService.upload(file, influencerId);
      if (!up.success || !up.documents_id) throw new Error(up.message || "업로드 실패");
      setMsg("업로드 완료, 벡터화 진행 중...");
      await DocumentService.vectorize(up.documents_id);
      setMsg("✅ 업로드 및 벡터화 완료");
      await load();
    } catch (e: any) {
      setMsg(`❌ ${e?.message || "처리 실패"}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleVectorize = async (id: string) => {
    setBusyId(id);
    setMsg("");
    try {
      await DocumentService.vectorize(id);
      setMsg("✅ 벡터화 완료");
      await load();
    } catch (e: any) {
      setMsg(`❌ 벡터화 실패: ${e?.message || ""}`);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("이 문서를 삭제할까요?")) return;
    setBusyId(id);
    try {
      await DocumentService.remove(id);
      await load();
    } catch (e: any) {
      setMsg(`❌ 삭제 실패: ${e?.message || ""}`);
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="p-6">
      <div className="text-center mb-6">
        <BookOpen className="h-8 w-8 mx-auto mb-2 text-blue-500" />
        <h2 className="text-xl font-bold mb-1">문서 / 지식 (RAG)</h2>
        <p className="text-gray-600 text-sm">
          PDF/TXT/MD 문서를 업로드하면 벡터화되어, 챗봇이 해당 문서를 근거로 답변합니다.
        </p>
      </div>

      {/* 업로드 영역 */}
      <div className="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center mb-4">
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleUpload(f);
          }}
        />
        <Button
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          className="bg-blue-500 hover:bg-blue-600"
        >
          {uploading ? (
            <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> 처리 중...</>
          ) : (
            <><Upload className="h-4 w-4 mr-2" /> 문서 업로드</>
          )}
        </Button>
        <p className="text-xs text-gray-400 mt-2">PDF / TXT / MD, 최대 10MB</p>
        {msg && <p className="text-sm mt-3">{msg}</p>}
      </div>

      {/* 문서 목록 */}
      {loading ? (
        <div className="text-center py-8 text-gray-500">
          <Loader2 className="h-6 w-6 animate-spin mx-auto" />
        </div>
      ) : docs.length === 0 ? (
        <div className="text-center py-8 text-gray-400 text-sm">업로드된 문서가 없습니다.</div>
      ) : (
        <div className="space-y-2">
          {docs.map((d) => (
            <div
              key={d.documents_id}
              className="flex items-center justify-between border rounded-lg px-4 py-3"
            >
              <div className="flex items-center space-x-3 min-w-0">
                {d.is_vectorized === 1 ? (
                  <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0" />
                ) : (
                  <Circle className="h-5 w-5 text-gray-300 shrink-0" />
                )}
                <div className="min-w-0">
                  <div className="font-medium text-sm truncate">{d.documents_name}</div>
                  <div className="text-xs text-gray-400">
                    {d.is_vectorized === 1 ? "벡터화 완료" : "미벡터화"}
                    {d.file_size ? ` · ${(d.file_size / 1024).toFixed(1)} KB` : ""}
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2 shrink-0">
                {d.is_vectorized !== 1 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleVectorize(d.documents_id)}
                    disabled={busyId === d.documents_id}
                  >
                    {busyId === d.documents_id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      "벡터화"
                    )}
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(d.documents_id)}
                  disabled={busyId === d.documents_id}
                >
                  <Trash2 className="h-4 w-4 text-red-400" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
