import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ threadId: string }> },
) {
  try {
    const { threadId } = await params;
    const headers: HeadersInit = {};
    const apiKey = process.env.RAG_API_KEY;
    if (apiKey) headers.authorization = `Bearer ${apiKey}`;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);

    try {
      const response = await fetch(
        `${process.env.BACKEND_URL || "http://localhost:8000"}/memory/${encodeURIComponent(threadId)}`,
        { headers, cache: "no-store", signal: controller.signal },
      );
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
    return NextResponse.json({ messages: [] }, { status: 503 });
  }
}
