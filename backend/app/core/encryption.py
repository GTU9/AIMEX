"""
AES256 암호화 유틸리티
허깅페이스 토큰 등 민감한 정보를 안전하게 암호화/복호화
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AESEncryption:
    def __init__(self, password: Optional[str] = None, salt: Optional[str] = None):
        """
        AES256 암호화 클래스 초기화

        키/솔트는 반드시 환경변수(ENCRYPTION_KEY/ENCRYPTION_SALT) 또는 인자로
        주입해야 한다. 하드코딩 기본값은 소스가 공개되면 암호화가 무력화되므로
        제공하지 않으며, 미설정 시 즉시 예외를 던진다(fail-fast).

        Args:
            password: 암호화 패스워드. 미지정 시 환경변수 ENCRYPTION_KEY 사용.
            salt: 솔트. 미지정 시 환경변수 ENCRYPTION_SALT 사용.
        """
        resolved_password = password or os.getenv('ENCRYPTION_KEY')
        if not resolved_password:
            raise RuntimeError(
                "ENCRYPTION_KEY 가 설정되지 않았습니다. 민감정보 암호화를 위해 "
                "강한 랜덤 키를 환경변수(.env)에 설정하세요. "
                "(예: python -c \"import secrets;print(secrets.token_urlsafe(48))\")"
            )

        resolved_salt = salt or os.getenv('ENCRYPTION_SALT')
        if not resolved_salt:
            raise RuntimeError(
                "ENCRYPTION_SALT 가 설정되지 않았습니다. 환경변수(.env)에 "
                "고유한 솔트 값을 설정하세요."
            )

        self.password = resolved_password
        self.salt = resolved_salt.encode() if isinstance(resolved_salt, str) else resolved_salt

        # PBKDF2를 사용하여 키 생성
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        self.fernet = Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        문자열을 암호화
        Args:
            plaintext: 암호화할 평문
        Returns:
            Base64로 인코딩된 암호화된 문자열
        """
        try:
            if not plaintext:
                return ""
            
            # 문자열을 바이트로 변환 후 암호화
            encrypted_bytes = self.fernet.encrypt(plaintext.encode('utf-8'))
            
            # Base64로 인코딩하여 문자열로 반환
            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"암호화 실패: {e}")
            raise Exception(f"암호화 중 오류가 발생했습니다: {e}")
    
    def decrypt(self, encrypted_text: str) -> str:
        """
        암호화된 문자열을 복호화
        Args:
            encrypted_text: Base64로 인코딩된 암호화된 문자열
        Returns:
            복호화된 평문 문자열
        """
        try:
            if not encrypted_text:
                return ""
            
            # Base64 디코딩
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))
            
            # 복호화 후 문자열로 변환
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"복호화 실패: {e}")
            raise Exception(f"복호화 중 오류가 발생했습니다: {e}")
    
    def is_encrypted(self, text: str) -> bool:
        """
        문자열이 암호화된 것인지 확인
        Args:
            text: 확인할 문자열
        Returns:
            암호화된 문자열이면 True, 아니면 False
        """
        try:
            if not text:
                return False
            
            # 복호화를 시도해보고 성공하면 암호화된 것으로 판단
            self.decrypt(text)
            return True
        except:
            return False


# 전역 암호화 인스턴스
_encryption_instance = None


def get_encryption() -> AESEncryption:
    """암호화 인스턴스 싱글톤 패턴으로 가져오기"""
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = AESEncryption()
    return _encryption_instance


def encrypt_sensitive_data(plaintext: str) -> str:
    """민감한 데이터 암호화 헬퍼 함수"""
    encryption = get_encryption()
    return encryption.encrypt(plaintext)


def decrypt_sensitive_data(encrypted_text: str) -> str:
    """민감한 데이터 복호화 헬퍼 함수"""
    if not encrypted_text:
        return ""
    
    try:
        encryption = get_encryption()
        return encryption.decrypt(encrypted_text)
    except Exception as e:
        logger.warning(f"복호화 실패: {e}")
        # 복호화 실패 시 빈 문자열 반환 (넘어감)
        return ""


def test_encryption():
    """암호화/복호화 테스트 함수"""
    try:
        encryption = AESEncryption()
        
        # 테스트 데이터
        test_data = [
            "hf_1234567890abcdef",
            "test-token-value",
            "허깅페이스-토큰-테스트",
            "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        ]
        
        print("=== AES256 암호화/복호화 테스트 ===")
        
        for original in test_data:
            print(f"\n원본: {original}")
            
            # 암호화
            encrypted = encryption.encrypt(original)
            print(f"암호화: {encrypted}")
            
            # 복호화
            decrypted = encryption.decrypt(encrypted)
            print(f"복호화: {decrypted}")
            
            # 검증
            if original == decrypted:
                print("✅ 성공")
            else:
                print("❌ 실패")
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"테스트 실패: {e}")


if __name__ == "__main__":
    test_encryption()