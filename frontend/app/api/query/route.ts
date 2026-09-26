import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const backendUrl = () => process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.text();
    const headers: HeadersInit = { "content-type": "application/json" };
    const apiKey = process.env.RAG_API_KEY;
    if (apiKey) headers.authorization = `Bearer ${apiKey}`;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60_000);

    try {
      const response = await fetch(`${backendUrl()}/query`, {
        method: "POST",
        headers,
        body,
        cache: "no-store",
        signal: controller.signal,
      });

      return new NextResponse(response.body, {
        status: response.status,
        headers: {
          "content-type":
            response.headers.get("content-type") || "application/json",
        },
      });
    } finally {
      clearTimeout(timeout);
    }
  } catch {
    return NextResponse.json(
      { detail: "RAG backend unavailable" },
      { status: 503 },
    );
  }
}
