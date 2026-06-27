import { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Brain, Cpu, Database, RefreshCw, Play, ExternalLink, CheckCircle2, Loader2, AlertCircle } from "lucide-react";
import { ModelService } from "@/lib/services/model.service";

interface FineTuningTabProps {
  influencerId: string;
  learningStatus?: number; // 0: 학습 중, 1: 사용 가능
}

// 진행 중으로 간주하는 상태들
const RUNNING = new Set([
  "pending", "tone_generation", "domain_preparation", "processing",
  "batch_submitted", "batch_processing", "batch_upload", "processing_results",
  "training", "running", "queued", "uploading",
]);

const STATUS_LABEL: Record<string, string> = {
  pending: "대기 중",
  tone_generation: "말투 생성 중",
  domain_preparation: "도메인 준비 중",
  processing: "처리 중",
  batch_submitted: "배치 제출됨",
  batch_processing: "배치 처리 중",
  batch_upload: "배치 업로드 중",
  processing_results: "결과 처리 중",
  training: "학습 중",
  running: "실행 중",
  queued: "대기열",
  uploading: "업로드 중",
  completed: "완료",
  failed: "실패",
  error: "오류",
};

function StatusBadge({ status }: { status: string }) {
  const s = (status || "").toLowerCase();
  const label = STATUS_LABEL[s] || status || "-";
  if (s === "completed") return <Badge className="bg-green-100 text-green-700 hover:bg-green-100">{label}</Badge>;
  if (s === "failed" || s === "error") return <Badge className="bg-red-100 text-red-700 hover:bg-red-100">{label}</Badge>;
  if (RUNNING.has(s)) return <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-100">{label}</Badge>;
  return <Badge variant="secondary">{label}</Badge>;
}

export default function FineTuningTab({ influencerId, learningStatus }: FineTuningTabProps) {
  const [qaTasks, setQaTasks] = useState<any[]>([]);
  const [ftTasks, setFtTasks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async () => {
    try {
      const [qa, ft] = await Promise.all([
        ModelService.getQaStatus(influencerId).catch(() => ({ tasks: [] })),
        ModelService.getFinetuningStatus(influencerId).catch(() => ({ finetuning_tasks: [] })),
      ]);
      setQaTasks(Array.isArray(qa?.tasks) ? qa.tasks : []);
      setFtTasks(Array.isArray(ft?.finetuning_tasks) ? ft.finetuning_tasks : []);
    } finally {
      setLoading(false);
    }
  }, [influencerId]);

  useEffect(() => { load(); }, [load]);

  // 진행 중 작업이 있으면 5초마다 자동 새로고침
  const anyRunning =
    qaTasks.some((t) => RUNNING.has(String(t.status).toLowerCase())) ||
    ftTasks.some((t) => RUNNING.has(String(t.status).toLowerCase()));

  useEffect(() => {
    if (anyRunning && !pollRef.current) {
      pollRef.current = setInterval(load, 5000);
    } else if (!anyRunning && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [anyRunning, load]);

  const handleStart = async () => {
    setTriggering(true);
    setMessage(null);
    try {
      const res = await ModelService.triggerQaGeneration(influencerId);
      setMessage(res?.message || "파인튜닝 파이프라인이 시작되었습니다.");
      await load();
    } catch (e: any) {
      setMessage(e?.message || "시작에 실패했습니다. 잠시 후 다시 시도해주세요.");
    } finally {
      setTriggering(false);
    }
  };

  const fmt = (d?: string) => (d ? new Date(d).toLocaleString("ko-KR") : "-");

  return (
    <div className="space-y-6">
      {/* 과정 안내 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">파인튜닝 과정</CardTitle>
          <p className="text-sm text-gray-600">
            이 인플루언서만의 말투·성격을 학습시키는 3단계 파이프라인입니다.
          </p>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-100 bg-amber-50 px-3 py-2.5 text-sm text-amber-800">
            <AlertCircle className="h-4 w-4 mt-0.5 text-amber-500 shrink-0" />
            <span>
              이 인플루언서는 <b>생성 시 한 번 자동으로 학습</b>되었습니다. 아래 "파인튜닝 시작"은
              <b> 설정·말투를 바꿨거나, 학습 문서/데이터를 추가했거나, 답변 품질을 더 높이고 싶을 때 다시 학습</b>하는 용도입니다.
              (재학습하면 새 LoRA 어댑터로 교체됩니다.)
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border border-gray-100 bg-gray-50 p-4">
              <div className="flex items-center gap-2 text-blue-600 mb-1">
                <Database className="h-4 w-4" /><span className="font-semibold text-sm">1. QA 데이터 생성</span>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">
                성격·말투 설정을 바탕으로 OpenAI 배치 API가 학습용 대화 데이터(QA)를 대량 생성합니다.
              </p>
            </div>
            <div className="rounded-lg border border-gray-100 bg-gray-50 p-4">
              <div className="flex items-center gap-2 text-purple-600 mb-1">
                <Cpu className="h-4 w-4" /><span className="font-semibold text-sm">2. LoRA 파인튜닝</span>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">
                생성된 QA로 Modal GPU에서 베이스 모델(EXAONE-3.5-2.4B)에 LoRA 어댑터를 학습하고 허깅페이스에 업로드합니다.
              </p>
            </div>
            <div className="rounded-lg border border-gray-100 bg-gray-50 p-4">
              <div className="flex items-center gap-2 text-green-600 mb-1">
                <Brain className="h-4 w-4" /><span className="font-semibold text-sm">3. 챗봇 반영</span>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">
                학습된 어댑터가 챗봇·테스트·외부 API에 자동 적용되어 고유한 캐릭터로 대화합니다.
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between mt-5">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-gray-500">현재 학습 상태:</span>
              {learningStatus === 1 ? (
                <Badge className="bg-green-100 text-green-700 hover:bg-green-100">사용 가능</Badge>
              ) : (
                <Badge className="bg-blue-100 text-blue-700 hover:bg-blue-100">학습 중</Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={load} disabled={loading}>
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />새로고침
              </Button>
              <Button size="sm" onClick={handleStart} disabled={triggering || anyRunning} className="bg-blue-500 hover:bg-blue-600">
                {triggering ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                {anyRunning ? "진행 중..." : (learningStatus === 1 ? "재학습 시작" : "파인튜닝 시작")}
              </Button>
            </div>
          </div>
          {message && (
            <div className="mt-3 flex items-start gap-2 text-sm text-gray-600 bg-blue-50 border border-blue-100 rounded-lg px-3 py-2">
              <AlertCircle className="h-4 w-4 mt-0.5 text-blue-500 shrink-0" /><span>{message}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* QA 생성 작업 */}
      <Card>
        <CardHeader><CardTitle className="text-base">QA 데이터 생성 작업</CardTitle></CardHeader>
        <CardContent>
          {qaTasks.length === 0 ? (
            <p className="text-sm text-gray-400 py-4 text-center">아직 QA 생성 작업이 없습니다. “파인튜닝 시작”으로 진행하세요.</p>
          ) : (
            <div className="space-y-2">
              {qaTasks.map((t, i) => (
                <div key={t.task_id || i} className="flex items-center justify-between rounded-lg border border-gray-100 px-3 py-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <StatusBadge status={t.status} />
                      <span className="text-xs text-gray-400 truncate">{fmt(t.created_at)}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      생성 QA: <b>{t.generated_qa_pairs ?? 0}</b>
                      {t.total_qa_pairs ? ` / ${t.total_qa_pairs}` : ""}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 파인튜닝 작업 */}
      <Card>
        <CardHeader><CardTitle className="text-base">LoRA 파인튜닝 작업</CardTitle></CardHeader>
        <CardContent>
          {ftTasks.length === 0 ? (
            <p className="text-sm text-gray-400 py-4 text-center">아직 파인튜닝 작업이 없습니다.</p>
          ) : (
            <div className="space-y-2">
              {ftTasks.map((t, i) => (
                <div key={t.task_id || i} className="flex items-center justify-between rounded-lg border border-gray-100 px-3 py-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <StatusBadge status={t.status} />
                      {t.model_name && <span className="text-xs font-medium text-gray-700 truncate">{t.model_name}</span>}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {t.training_epochs ? `epoch ${t.training_epochs} · ` : ""}{fmt(t.created_at)}
                    </p>
                  </div>
                  {t.hf_model_url || t.hf_repo_id ? (
                    <a
                      href={t.hf_model_url || `https://huggingface.co/${t.hf_repo_id}`}
                      target="_blank" rel="noreferrer"
                      className="flex items-center gap-1 text-xs text-blue-600 hover:underline shrink-0"
                    >
                      {t.status?.toLowerCase() === "completed" && <CheckCircle2 className="h-3.5 w-3.5" />}
                      모델 보기 <ExternalLink className="h-3 w-3" />
                    </a>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
