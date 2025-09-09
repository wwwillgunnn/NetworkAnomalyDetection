"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Bell, Search } from "lucide-react";

export default function DashboardPage() {
  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-md flex flex-col">
        <div className="p-4 text-xl font-bold border-b">Anomaly Dash</div>
        <nav className="flex flex-col gap-2 p-4">
          <Button variant="ghost" className="justify-start">
            📊 Overview
          </Button>
          <Button variant="ghost" className="justify-start">
            🌐 Live Traffic
          </Button>
          <Button variant="ghost" className="justify-start">
            ⚠️ Anomalies
          </Button>
          <Button variant="ghost" className="justify-start">
            ⚙️ Settings
          </Button>
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {/* Topbar */}
        <header className="flex items-center justify-between bg-white px-6 py-4 shadow-sm">
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
              <Input placeholder="Search..." className="pl-8" />
            </div>
            <Button variant="ghost" size="icon">
              <Bell className="h-5 w-5" />
            </Button>
            <Avatar>
              <AvatarFallback>WG</AvatarFallback>
            </Avatar>
          </div>
        </header>

        {/* Metrics */}
        <main className="flex-1 p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Total Packets</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">12,340</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Anomalies Detected</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-red-500">47</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold text-green-600">96%</p>
              </CardContent>
            </Card>
          </div>

          {/* Placeholder for charts */}
          <Card>
            <CardHeader>
              <CardTitle>Traffic Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center text-gray-400">
                [Chart goes here 📈]
              </div>
            </CardContent>
          </Card>

          {/* Table */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Anomalies</CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="text-left border-b">
                    <th className="py-2">Timestamp</th>
                    <th className="py-2">Source IP</th>
                    <th className="py-2">Destination IP</th>
                    <th className="py-2">Severity</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b">
                    <td className="py-2">2025-09-09 10:33</td>
                    <td>192.168.1.15</td>
                    <td>10.0.0.45</td>
                    <td className="text-red-500 font-medium">High</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-2">2025-09-09 10:35</td>
                    <td>172.16.0.8</td>
                    <td>10.0.0.10</td>
                    <td className="text-yellow-500 font-medium">Medium</td>
                  </tr>
                  <tr>
                    <td className="py-2">2025-09-09 10:40</td>
                    <td>192.168.1.20</td>
                    <td>10.0.0.50</td>
                    <td className="text-green-600 font-medium">Low</td>
                  </tr>
                </tbody>
              </table>
            </CardContent>
          </Card>
        </main>
      </div>
    </div>
  );
}
