"use client";

// Instagram 연동/업로드 기능은 현재 비활성화되어 있습니다.
// 외부 요건(인스타그램 비즈니스 계정 + 연결된 Facebook 페이지, Meta 앱 권한/리뷰,
// 공개 접근 가능한 이미지 URL)이 갖춰지면 복구합니다.
// 백엔드 연동/게시 코드는 보존되어 있어 이 페이지만 되돌리면 재활성화됩니다.

import { Instagram } from "lucide-react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

export default function InstagramDisabledPage() {
  const router = useRouter();
  return (
    <div className="min-h-[60vh] flex items-center justify-center p-6">
      <div className="text-center max-w-md">
        <Instagram className="w-12 h-12 mx-auto mb-4 text-gray-300" />
        <h1 className="text-xl font-bold mb-2">Instagram 연동 (준비 중)</h1>
        <p className="text-gray-600 text-sm mb-6">
          인스타그램 업로드 기능은 현재 비활성화되어 있습니다. 비즈니스 계정 연동 요건이
          준비되면 제공될 예정입니다.
        </p>
        <Button onClick={() => router.push("/dashboard")}>대시보드로 이동</Button>
      </div>
    </div>
  );
}
