# UAT Checklist (User + Model Acceptance)

Use this checklist before sign-off for a release candidate.

## 1) Test Setup
- [ ] Candidate build/version is identified.
- [ ] Test environment matches target deployment settings.

## 2) Core User Flow Acceptance
- [ ] User can submit a generation job with valid inputs.
- [ ] Input validation errors are clear and actionable.
- [ ] Job status updates are visible and accurate.
- [ ] User can download completed outputs successfully.
- [ ] Error states show understandable messages and recovery guidance.

## 3) API/Platform Acceptance
- [ ] Health endpoint returns expected status.
- [ ] Auth/ownership checks prevent cross-user access.
- [ ] Invalid/nonexistent job IDs return correct status codes.
- [ ] Presigned download flow works and expires as expected.

## 4) Model Acceptance Criteria
- [ ] Model run completes for representative tickers/assets.
- [ ] Validation metrics are generated (KS, kurtosis, ACF, etc.).
- [ ] Metrics are within agreed acceptance thresholds.
- [ ] Baseline comparison is recorded (AI vs fixed/heuristic).
- [ ] Any model warnings/failures are reviewed and documented.

## 5) Non-Functional Acceptance
- [ ] End-to-end runtime is within acceptable limits.
- [ ] No critical errors in service logs during test window.
- [ ] Output file format/schema matches consumer expectations.
- [ ] Security/privacy checks completed for exposed endpoints.

## 6) Defects and Sign-Off
- [ ] All critical/high defects are resolved or explicitly waived.
- [ ] Medium/low defects are triaged with owners and timelines.
- [ ] Final test results are attached (`TEST_RESULTS.md`).
- [ ] Product/engineering sign-off recorded.

Sign-off:
- UAT Lead:
- Product Owner:
- Engineering Owner:
- Date:
