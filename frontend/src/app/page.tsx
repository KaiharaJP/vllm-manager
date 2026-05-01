import AuthGate from "@/components/AuthGate";
import Dashboard from "@/components/Dashboard";

export default function Home() {
  return (
    <main className="min-h-screen">
      <AuthGate>
        {(user, logout) => <Dashboard currentUser={user} onLogout={logout} />}
      </AuthGate>
    </main>
  );
}
