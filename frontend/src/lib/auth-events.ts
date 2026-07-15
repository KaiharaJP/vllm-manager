/** backend 再起動などでセッション JWT が失効したときに発火する。 */
export const AUTH_EXPIRED_EVENT = "vllm-manager:auth-expired";

export function notifyAuthExpired() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT));
}
