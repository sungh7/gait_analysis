# 진행 상황 요약

- `npm install`로 프로젝트 의존성을 설치하고 Next.js 개발 서버 실행 여부를 확인했습니다.
- `app/page.tsx`에서 `yearInfo`가 초기화되기 전에 접근되던 버그를 수정하여 최초 렌더링 시 ReferenceError가 더 이상 발생하지 않습니다.
- 네트워크 데이터를 불러올 때 로딩 스피너가 바로 사라지지 않도록 `showNetworkLoading` 상태를 추가하고, API 오류 메시지를 사용자에게 직접 노출하도록 개선했습니다.
- `lib/semanticScholar.ts`에 재시도 로직, API Key 헤더, 사용자 지정 User-Agent, 의미 있는 오류 메시지를 추가해 Semantic Scholar API 응답 안정성을 높였습니다.
- `app/api/network/[id]/route.ts`에서 null-safe 검사를 적용하고 SemanticScholarApiError를 구분 처리하여 서버 500 오류를 줄였습니다.
- README에 `SEMANTIC_SCHOLAR_API_KEY` 사용법을 문서화하여 레이트 리밋 회피 방법을 안내했습니다.

# 현재 문제점 및 리스크

- API Key 없이 테스트할 경우 Semantic Scholar API의 429(too many requests) 응답이 자주 발생하며, 이는 여전히 데이터 로딩 지연의 주된 원인입니다.
- 네트워크 데이터가 매번 외부 API 호출에 의존하면서 로딩 시간이 길고, 반복 조회 시에도 캐시가 없어 사용 경험이 저하됩니다.
- Vector 기반 추천을 위한 로컬 데이터베이스/RAG 파이프라인이 아직 구축되지 않아, 향후 확장성 및 성능 측면에서 한계가 존재합니다.

# 다음 단계 제안

- Semantic Scholar API Key와 User-Agent를 환경 변수에 설정하여 레이트 리밋을 완화하십시오.
- SQLite + Prisma/Drizzle 등을 활용한 로컬 캐시 계층을 도입해 자주 조회되는 논문 네트워크를 DB에서 바로 반환하도록 구현하면 로딩 시간을 크게 줄일 수 있습니다.
- 향후 RAG 추천 품질 개선을 위해 논문 메타데이터와 임베딩을 저장하는 벡터 데이터베이스(예: sqlite-vec, Chroma, LanceDB) 연동을 검토하십시오.
