import { getJsonAuth, postJsonAuth } from "../../http";

export type AdminAuthResponse = {
  authenticated: boolean;
  expires_at: number | null;
};

export function adminSession(): Promise<AdminAuthResponse> {
  return getJsonAuth<AdminAuthResponse>("/api/admin/auth/session");
}

export function adminLogin(password: string): Promise<AdminAuthResponse> {
  return postJsonAuth<AdminAuthResponse>("/api/admin/auth/login", { password });
}

export function adminLogout(): Promise<AdminAuthResponse> {
  return postJsonAuth<AdminAuthResponse>("/api/admin/auth/logout");
}
