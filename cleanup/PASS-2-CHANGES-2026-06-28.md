# Cleanup pass 2 changes and reasons

## Changed

1. Authentication utilities no longer print token fragments or token metadata.
   - Reason: console output is observable by users and extensions and is not required for authentication behavior.
2. Post creation and post detail flows no longer print user-generated content, file objects, user objects, or API results.
   - Reason: these logs increase privacy exposure, browser noise, and serialization cost without affecting control flow.
3. Image-generation proxy routes no longer print request bodies.
   - Reason: prompts and generation parameters can be sensitive and server logs should not receive them by default.
4. Chat avatar rendering and gallery rendering no longer execute console side effects.
   - Reason: render functions should remain side-effect free and may run repeatedly under React development checks.
5. Tone generation and fine-tuning code no longer emit ad-hoc raw debug output.
   - Reason: operational output should use structured logging with an intentional level and redaction policy.

## Not changed

Potentially behavior-changing fallback, authorization, migration, dependency, and decomposition work remains documented as issues. Removing those paths without targeted regression coverage would turn a cleanup pass into an unreviewable product change.
