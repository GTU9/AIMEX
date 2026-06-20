import { Eye, EyeOff, Copy, RefreshCw, Send, Loader2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ModelService } from "@/lib/services/model.service";

interface ApiKeyInfo {
  created_at: string;
  updated_at: string;
}

interface Model {
  apiKey?: string;
}

interface ApiTabProps {
  model: Model;
  showApiKey: boolean;
  setShowApiKey: (show: boolean) => void;
  isGeneratingApiKey: boolean;
  apiKeyInfo: ApiKeyInfo | null;
  copyApiKey: () => void;
  generateNewApiKey: () => void;
}

export default function ApiTab({
  model,
  showApiKey,
  setShowApiKey,
  isGeneratingApiKey,
  apiKeyInfo,
  copyApiKey,
  generateNewApiKey
}: ApiTabProps) {
  const [testMessage, setTestMessage] = useState("안녕! 간단히 자기소개 해줘.");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testError, setTestError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

  const handleTest = async () => {
    if (!model.apiKey || !testMessage.trim()) return;
    setTesting(true);
    setTestResult(null);
    setTestError(null);
    try {
      const res = await ModelService.callChatbot(model.apiKey, { message: testMessage });
      setTestResult(res.response);
    } catch (e: any) {
      setTestError(e?.message || "호출에 실패했습니다. 모델 상태를 확인해주세요.");
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>API 키 관리</CardTitle>
          <CardDescription>
            AI 모델에 접근하기 위한 API 키를 관리합니다
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="api-key">API 키</Label>
            <div className="flex space-x-2">
              <Input
                id="api-key"
                type={showApiKey ? "text" : "password"}
                value={model.apiKey || ""}
                readOnly
                className="font-mono"
                placeholder={
                  isGeneratingApiKey
                    ? "생성 중..."
                    : "API 키를 불러오는 중..."
                }
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowApiKey(!showApiKey)}
              >
                {showApiKey ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={copyApiKey}
              >
                <Copy className="h-4 w-4" />
              </Button>
            </div>
            {apiKeyInfo && (
              <div className="mt-2 text-xs text-gray-500 space-y-1">
                <div>
                  생성일:{" "}
                  {new Date(apiKeyInfo.created_at).toLocaleDateString(
                    "ko-KR",
                  )}
                </div>
                {apiKeyInfo.updated_at !== apiKeyInfo.created_at && (
                  <div>
                    수정일:{" "}
                    {new Date(apiKeyInfo.updated_at).toLocaleDateString(
                      "ko-KR",
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              onClick={generateNewApiKey}
              disabled={isGeneratingApiKey}
            >
              {isGeneratingApiKey ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  생성 중...
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />새 키 생성
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API 테스트</CardTitle>
          <CardDescription>
            발급된 API 키로 챗봇 엔드포인트를 실제 호출해 응답을 확인합니다
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex space-x-2">
            <Input
              value={testMessage}
              onChange={(e) => setTestMessage(e.target.value)}
              placeholder="테스트 메시지를 입력하세요"
              onKeyDown={(e) => { if (e.key === "Enter") handleTest(); }}
            />
            <Button onClick={handleTest} disabled={testing || !model.apiKey} className="bg-blue-500 hover:bg-blue-600 shrink-0">
              {testing ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Send className="h-4 w-4 mr-2" />}
              테스트 호출
            </Button>
          </div>
          {testResult && (
            <div className="rounded-md border border-blue-100 bg-blue-50 p-3">
              <div className="text-xs font-semibold text-blue-700 mb-1">응답</div>
              <p className="text-sm text-gray-800 whitespace-pre-wrap">{testResult}</p>
            </div>
          )}
          {testError && (
            <div className="rounded-md border border-red-100 bg-red-50 p-3 text-sm text-red-600">{testError}</div>
          )}
          {testing && !testResult && (
            <p className="text-xs text-gray-500">모델 호출 중… (콜드스타트 시 시간이 걸릴 수 있어요)</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API 사용법</CardTitle>
          <CardDescription>
            AI 모델을 호출하는 방법을 안내합니다
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <Label>단일 응답 엔드포인트</Label>
              <div className="bg-gray-100 p-3 rounded-md font-mono text-sm">
                POST https://api.aimex.toyproject.site/v1/chat/chatbot
              </div>
            </div>
            <div>
              <Label>요청 예시</Label>
              <pre className="bg-gray-100 p-3 rounded-md text-sm overflow-x-auto">
                {`curl -X POST https://api.aimex.toyproject.site/v1/chat/chatbot \\
                  -H "Authorization: Bearer [API_KEY]" \\
                  -H "Content-Type: application/json" \\
                  -d '{
                    "message": "안녕하세요! 오늘 패션 추천 부탁드려요"
                  }'`}
              </pre>
            </div>
          </div>
          <div className="space-y-4 mt-4">
            <div>
              <Label>스트리밍 엔드포인트</Label>
              <div className="bg-gray-100 p-3 rounded-md font-mono text-sm">
                POST https://api.aimex.toyproject.site/v1/chat/chatbot/stream
              </div>
            </div>
            <div>
              <Label>요청 예시</Label>
              <pre className="bg-gray-100 p-3 rounded-md text-sm overflow-x-auto">
                {`curl -X POST https://api.aimex.toyproject.site/v1/chat/chatbot/stream \\
                  -H "Authorization: Bearer [API_KEY]" \\
                  -H "Content-Type: application/json" \\
                  -d '{
                    "message": "안녕하세요! 오늘 패션 추천 부탁드려요"
                  }'`}
              </pre>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}