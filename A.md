# 🚀 VortexFlow AI — Master Build Roadmap Prompt
### For Google AI Studio (Gemini) → React + TypeScript + Vite

---

## 📌 HOW TO USE THIS FILE

1. Open **Google AI Studio** → **Build** tab
2. Copy the **current milestone's prompt block** (marked with `### 🔨 PASTE THIS`)
3. Paste it into AI Studio and let it generate
4. Preview the output in the live preview panel
5. When satisfied, type **"next"** to move to the next milestone
6. Repeat until the full app is built

---

## 🧠 PROJECT OVERVIEW

**Project Name:** VortexFlow AI  
**Type:** Commercial & Professional AI Chatbot Web Application  
**Stack:** React 18 + TypeScript + Vite + Firebase (Auth + RTDB) + Google Gemini API  
**Auth:** Firebase Email/Password + Google OAuth  
**Database:** Firebase Realtime Database (RTDB)  
**AI Backend:** Google Gemini API (gemini-1.5-flash / gemini-1.5-pro)  
**Styling:** Tailwind CSS + Framer Motion  
**No fake/mock data — all real, realtime Firebase data**  
**Free tier only — no payments, no Stripe, no subscription walls**

---

## 🏗️ ARCHITECTURE OVERVIEW

```
VortexFlow AI
├── Landing Page (/)
├── Auth Pages (/login, /signup, /forgot-password)
├── Main Chat App (/chat)
│   ├── Sidebar (conversation history, new chat, search)
│   ├── Chat Window (messages, streaming, markdown)
│   ├── Input Bar (text, voice input UI, file attach UI)
│   └── Header (user menu, model selector, settings)
├── Modals
│   ├── Settings Modal (theme, language, API config)
│   ├── Profile Modal (edit name, avatar, bio)
│   ├── Keyboard Shortcuts Modal
│   ├── Share Chat Modal
│   ├── Delete Conversation Modal
│   ├── Clear All Chats Modal
│   ├── Export Chat Modal (JSON/TXT/MD)
│   ├── Feedback Modal
│   ├── About Modal
│   └── New Chat Confirmation Modal
└── 404 Page
```

---

## ⚙️ FIREBASE CONFIG (Use this in every milestone that needs Firebase)

```typescript
// src/lib/firebase.ts
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getDatabase } from "firebase/database";

const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  databaseURL: "YOUR_DATABASE_URL",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getDatabase(app);
export const googleProvider = new GoogleAuthProvider();
```

> ⚠️ User will replace placeholder values with real Firebase project credentials.

---

## 📦 DEPENDENCIES TO INSTALL

```bash
npm create vite@latest vortexflow-ai -- --template react-ts
cd vortexflow-ai
npm install firebase
npm install react-router-dom
npm install tailwindcss @tailwindcss/typography postcss autoprefixer
npm install framer-motion
npm install react-markdown remark-gfm rehype-highlight
npm install highlight.js
npm install react-hot-toast
npm install lucide-react
npm install @google/generative-ai
npm install react-intersection-observer
npm install date-fns
npm install uuid
npm install @types/uuid
npx tailwindcss init -p
```

---

## 🗄️ FIREBASE RTDB STRUCTURE

```json
{
  "users": {
    "$uid": {
      "displayName": "string",
      "email": "string",
      "photoURL": "string | null",
      "bio": "string",
      "createdAt": "timestamp",
      "lastSeen": "timestamp",
      "preferences": {
        "theme": "dark | light | system",
        "language": "en",
        "model": "gemini-1.5-flash | gemini-1.5-pro",
        "streamingEnabled": true,
        "soundEnabled": false,
        "fontSize": "sm | md | lg"
      },
      "stats": {
        "totalChats": 0,
        "totalMessages": 0,
        "joinedAt": "timestamp"
      }
    }
  },
  "conversations": {
    "$uid": {
      "$conversationId": {
        "id": "string",
        "title": "string",
        "createdAt": "timestamp",
        "updatedAt": "timestamp",
        "model": "gemini-1.5-flash",
        "messageCount": 0,
        "isPinned": false,
        "messages": {
          "$messageId": {
            "id": "string",
            "role": "user | assistant",
            "content": "string",
            "timestamp": "timestamp",
            "isError": false,
            "tokens": 0
          }
        }
      }
    }
  }
}
```

---

---

# ═══════════════════════════════════════════════
# MILESTONE 1 — Project Foundation & File Structure
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
You are building VortexFlow AI — a professional, commercial-grade AI Chatbot web application using React 18 + TypeScript + Vite + Tailwind CSS + Firebase + Google Gemini API. This is Milestone 1.

TASK: Create the complete project foundation with all configuration files, folder structure, global styles, theme system, and base utilities.

CREATE THE FOLLOWING FILES:

1. `vite.config.ts` — with path aliases (@/), env support
2. `tsconfig.json` — strict mode, path aliases
3. `tailwind.config.ts` — custom colors for VortexFlow brand (deep purples, electric blues, dark grays), dark/light mode via class, custom animations, custom fonts
4. `postcss.config.js`
5. `index.html` — proper meta tags, Open Graph, title "VortexFlow AI", favicon emoji ⚡, Google Fonts (Inter + JetBrains Mono)
6. `src/index.css` — Tailwind directives + global CSS variables for both dark/light themes, custom scrollbar styles, selection styles, base resets
7. `src/main.tsx` — App entry with React.StrictMode
8. `src/App.tsx` — Router setup with React Router v6, protected routes, lazy loading for all pages
9. `src/lib/firebase.ts` — Firebase init with Auth + RTDB + GoogleAuthProvider (with placeholder config)
10. `src/lib/gemini.ts` — Google Generative AI init, streamMessage function, generateTitle function
11. `src/store/` — Create these Zustand-like context stores using React Context + useReducer:
    - `AuthContext.tsx` — user state, loading, error
    - `ChatContext.tsx` — conversations, active conversation, messages, streaming state
    - `UIContext.tsx` — theme, sidebar open/close, active modal, toast queue
    - `PreferencesContext.tsx` — model selection, font size, sound, streaming toggle
12. `src/types/index.ts` — ALL TypeScript interfaces:
    - User, UserPreferences, UserStats
    - Conversation, Message, MessageRole
    - ModelOption, Theme, FontSize
    - ModalType enum
    - AppError
13. `src/utils/` — utility files:
    - `cn.ts` — classname merger utility
    - `formatDate.ts` — date formatting with date-fns
    - `generateId.ts` — UUID wrapper
    - `truncate.ts` — string truncation
    - `exportChat.ts` — export to JSON, TXT, MD formats
    - `generateChatTitle.ts` — extract title from first message
14. `src/hooks/` — custom hooks:
    - `useAuth.ts`
    - `useChat.ts`
    - `useTheme.ts`
    - `useModal.ts`
    - `useLocalStorage.ts`
    - `useDebounce.ts`
    - `useScrollToBottom.ts`
    - `useKeyboard.ts` — keyboard shortcut handler
15. `src/constants/index.ts` — APP_NAME, MODELS array with labels, KEYBOARD_SHORTCUTS object, MAX_MESSAGE_LENGTH, DEFAULT_PREFERENCES, SYSTEM_PROMPT for Gemini

BRAND COLORS for Tailwind config:
- Primary: #7C3AED (violet-600) → #9333EA (purple-600)
- Accent: #06B6D4 (cyan-500)
- Dark bg: #0A0A0F (near black)
- Dark surface: #111118
- Dark card: #1A1A26
- Dark border: #2A2A3A
- Light bg: #F8F7FF
- Light surface: #FFFFFF
- Text primary dark: #F0EDFF
- Text secondary dark: #9B97B0

SYSTEM PROMPT constant:
"You are VortexFlow AI, a highly capable, helpful, and professional AI assistant. You provide accurate, thoughtful, and well-structured responses. You support markdown formatting including code blocks, tables, lists, and more. Be conversational yet precise."

After creating all files, show a complete working src/ folder tree. The app should compile with `npm run dev` without errors. Show ALL file contents completely — do not truncate any file.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 2 — Landing Page (Full Professional)
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 2: Landing Page.

Using the foundation from Milestone 1 (React 18 + TypeScript + Vite + Tailwind CSS + Framer Motion), build a stunning, professional, fully animated landing page at route `/`.

CREATE: `src/pages/LandingPage.tsx` and all required sub-components in `src/components/landing/`

LANDING PAGE SECTIONS (in order):

1. **Navbar** (`LandingNavbar.tsx`)
   - Logo: "⚡ VortexFlow AI" with gradient text (purple to cyan)
   - Nav links: Features, How It Works, FAQ
   - CTA buttons: "Sign In" (ghost) and "Get Started Free" (gradient primary)
   - Glassmorphism background on scroll (backdrop-blur, border-bottom)
   - Hamburger menu for mobile, animated drawer
   - Sticky top, z-50
   - Smooth scroll to sections

2. **Hero Section** (`HeroSection.tsx`)
   - Large gradient headline: "The AI That Thinks\nWith You, Not For You"
   - Subheadline: "VortexFlow AI combines the power of Google Gemini with a beautiful, distraction-free interface. Have real conversations, get real answers."
   - Two CTA buttons: "Start Chatting Free →" (large gradient) + "See How It Works" (ghost)
   - Animated floating badge: "⚡ Powered by Google Gemini"
   - Hero visual: Animated mock chat window showing a sample conversation (hardcoded, looks real)
   - The mock chat shows messages typing in with a typewriter animation
   - Background: Dark with animated gradient orbs (purple + cyan), subtle grid pattern
   - Particle-like floating dots in background using CSS animation
   - Full viewport height

3. **Stats Bar** (`StatsBar.tsx`)
   - 4 animated counter stats: "10K+ Conversations", "99.9% Uptime", "< 1s Response", "100% Free"
   - Dark glassmorphism cards, gradient numbers
   - Animate counters up when in viewport (Intersection Observer)

4. **Features Section** (`FeaturesSection.tsx`)
   - Section title: "Everything You Need in an AI Assistant"
   - 6 feature cards in a 3x2 grid:
     * ⚡ Lightning Fast — Streaming responses powered by Gemini
     * 🧠 Context Aware — Remembers your full conversation
     * 📝 Rich Markdown — Code, tables, lists rendered beautifully
     * 🔒 Private & Secure — Firebase auth, your data stays yours
     * 💾 Chat History — All conversations saved and searchable
     * 🎨 Beautiful UI — Dark/light mode, customizable experience
   - Each card: icon, title, description, subtle hover glow animation
   - Staggered entrance animation with Framer Motion

5. **How It Works** (`HowItWorks.tsx`)
   - Section title: "Get Started in Seconds"
   - 3 steps with large step numbers, icons, title, description:
     1. Create Account — Sign up free with email or Google
     2. Start a Chat — Type anything, ask anything
     3. Get Answers — Streaming AI responses with full markdown
   - Connected by animated dashed line
   - Alternating layout on desktop

6. **Gemini Showcase** (`GeminiShowcase.tsx`)
   - "Powered by Google Gemini" section
   - Show model options: Gemini 1.5 Flash (fast, efficient) and Gemini 1.5 Pro (powerful, complex)
   - Two comparison cards showing speed vs power trade-off
   - Gemini branding colors subtly included

7. **FAQ Section** (`FAQSection.tsx`)
   - Section title: "Frequently Asked Questions"
   - 8 FAQ items as animated accordion:
     1. Is VortexFlow AI really free? — Yes, completely free. No credit card required.
     2. What AI model powers VortexFlow? — Google Gemini 1.5 (Flash and Pro)
     3. Is my data private? — Yes, stored securely in Firebase, only you can access your chats
     4. Can I export my conversations? — Yes, export as JSON, TXT, or Markdown
     5. Is there a message limit? — Fair use policy, no hard limits on free tier
     6. Does it support code? — Yes, full syntax highlighting for 100+ languages
     7. Can I use it on mobile? — Yes, fully responsive on all devices
     8. How do I sign in? — Email/password or Google OAuth, both supported
   - Smooth accordion open/close animation

8. **CTA Banner** (`CTABanner.tsx`)
   - Dark gradient background (purple to cyan gradient)
   - Headline: "Ready to Experience AI That Understands You?"
   - Subtext: "Join thousands of users. Free forever. No credit card needed."
   - Large "Start for Free →" button
   - Animated background glow

9. **Footer** (`LandingFooter.tsx`)
   - Logo + tagline
   - Links: Privacy Policy (placeholder), Terms (placeholder), Contact (placeholder)
   - "Built with ❤️ using Google Gemini API"
   - Copyright: © 2025 VortexFlow AI. All rights reserved.
   - Social icons (Twitter/X, GitHub) — placeholder links

REQUIREMENTS:
- Framer Motion for ALL section entrance animations (fadeInUp, staggerChildren)
- Fully responsive (mobile, tablet, desktop)
- Dark theme by default (matching overall app theme)
- All text, icons, colors consistent with brand (purple/cyan/dark)
- Smooth scroll behavior
- React Router Links for auth buttons pointing to /login and /signup
- No images — use emojis, icons (lucide-react), CSS gradients only
- Performance: use `loading="lazy"` and `will-change` where needed

Show ALL component files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 3 — Authentication Pages & Firebase Auth
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 3: Authentication System.

Build complete Firebase Authentication with Email/Password and Google OAuth. Create beautiful, professional auth pages.

CREATE THE FOLLOWING:

**Pages:**
- `src/pages/LoginPage.tsx`
- `src/pages/SignupPage.tsx`
- `src/pages/ForgotPasswordPage.tsx`

**Components in `src/components/auth/`:**
- `AuthLayout.tsx` — shared wrapper for all auth pages
- `AuthCard.tsx` — glassmorphism card container
- `EmailPasswordForm.tsx` — reusable form component
- `GoogleAuthButton.tsx` — "Continue with Google" button
- `AuthDivider.tsx` — "— or —" divider
- `PasswordStrengthBar.tsx` — animated password strength indicator
- `FormField.tsx` — reusable input with label, error, icon

**Auth Context (`src/store/AuthContext.tsx`) — FULL IMPLEMENTATION:**
```typescript
// Implement these functions using Firebase Auth:
- signInWithEmail(email, password)
- signUpWithEmail(email, displayName, password)
- signInWithGoogle()
- signOut()
- resetPassword(email)
- updateUserProfile(data)
- deleteAccount()
// On auth state change: sync user to Firebase RTDB under /users/$uid
// Store: displayName, email, photoURL, createdAt, lastSeen, preferences (defaults), stats
```

**LOGIN PAGE:**
- Left panel (desktop): Brand visual with animated logo, tagline, feature bullets
- Right panel: Login form
- Form fields: Email (with validation), Password (toggle show/hide)
- "Forgot password?" link
- Submit button with loading spinner
- Google OAuth button
- Link to signup
- Error messages shown as inline field errors + toast
- On success: redirect to /chat
- Remember me checkbox (stores in localStorage)

**SIGNUP PAGE:**
- Left panel: Same brand panel (different tagline)
- Right panel: Signup form
- Form fields: Full Name, Email, Password, Confirm Password
- Real-time password strength bar (Weak/Fair/Strong/Very Strong) based on:
  * Length ≥ 8
  * Has uppercase
  * Has number
  * Has special char
- Password requirements checklist (animated checkmarks as you type)
- Terms checkbox: "I agree to the Terms of Service and Privacy Policy"
- Submit button with loading
- Google OAuth button
- Link to login
- On success: Create user in Firebase RTDB, redirect to /chat with welcome toast

**FORGOT PASSWORD PAGE:**
- Simple centered card
- Email input
- Submit sends Firebase resetPassword email
- Success state: show "Check your inbox!" with animated mail icon
- Back to login link

**AUTH LAYOUT:**
- Full page, dark background with animated gradient orbs
- "⚡ VortexFlow AI" logo top-left links to /
- Animated entrance for card (slide up + fade)

**PROTECTED ROUTE (`src/components/routing/ProtectedRoute.tsx`):**
- Check AuthContext for user
- If loading: show full-page spinner with VortexFlow logo
- If not authenticated: redirect to /login with `state.from` for redirect back
- If authenticated: render children

**PUBLIC ROUTE (`src/components/routing/PublicRoute.tsx`):**
- If already authenticated: redirect to /chat
- Otherwise: render children (auth pages)

**FORM VALIDATION (no external library, custom):**
- Email: regex validation, required
- Password: min 8 chars, required
- Display name: min 2 chars, max 50 chars
- Real-time validation on blur, show errors on submit

**Firebase RTDB on signup:**
```typescript
// Write to /users/$uid on new account creation:
{
  displayName: name,
  email: email,
  photoURL: null,
  bio: "",
  createdAt: Date.now(),
  lastSeen: Date.now(),
  preferences: {
    theme: "dark",
    language: "en",
    model: "gemini-1.5-flash",
    streamingEnabled: true,
    soundEnabled: false,
    fontSize: "md"
  },
  stats: {
    totalChats: 0,
    totalMessages: 0,
    joinedAt: Date.now()
  }
}
```

STYLING:
- Auth card: glassmorphism (bg-white/5, backdrop-blur-xl, border border-white/10)
- Inputs: dark bg, subtle border, focus ring in brand purple, icon left side
- Buttons: gradient purple-to-cyan for primary, white/10 for Google
- All Framer Motion entrance animations
- Fully responsive — stacked on mobile, split on desktop

Show ALL files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 4 — Main Chat Layout & Sidebar
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 4: Main Chat Layout & Sidebar.

Build the main chat application shell — the layout, sidebar, and header that will house the chat experience.

CREATE:

**Pages:**
- `src/pages/ChatPage.tsx` — main layout orchestrator

**Components in `src/components/layout/`:**
- `AppLayout.tsx` — root layout with sidebar + main content
- `AppHeader.tsx` — top header bar
- `Sidebar.tsx` — left conversation list sidebar
- `SidebarHeader.tsx` — logo + new chat button
- `SidebarSearch.tsx` — search conversations input
- `ConversationList.tsx` — list of past conversations
- `ConversationItem.tsx` — single conversation row
- `ConversationContextMenu.tsx` — right-click menu (rename, pin, delete)
- `SidebarFooter.tsx` — user avatar + name + settings gear

**APP LAYOUT:**
- Desktop: Fixed sidebar (280px) + flex-1 main content, side by side
- Mobile: Sidebar as slide-in drawer (full height, overlay), toggle with hamburger
- Smooth sidebar open/close animation with Framer Motion
- Sidebar state persisted in localStorage

**SIDEBAR:**
- Background: Dark (#111118), left border
- Header:
  * Logo "⚡ VortexFlow AI" (small, gradient)
  * "New Chat" button (full width, gradient, sparkle icon)
- Search bar:
  * Searches through conversation titles in real-time
  * Debounced (300ms)
  * Shows "No results" state
- Conversation List:
  * Grouped by date: "Today", "Yesterday", "This Week", "Older"
  * Each item: conversation title (truncated), time ago, hover actions
  * Active conversation highlighted with purple left border + bg
  * Hover reveals: rename (pencil), pin (pin icon), delete (trash) buttons
  * Pinned conversations shown at top with 📌 icon
  * Loading skeleton shimmer while fetching from Firebase
  * Empty state: "No conversations yet. Start a new chat!"
  * Virtualized list (manual, using CSS for performance — only show visible items)
- Footer:
  * User avatar (circle with initials fallback if no photo)
  * Display name + email (truncated)
  * Gear icon → opens Settings modal

**APP HEADER:**
- Left: Hamburger menu (mobile) / sidebar toggle (desktop) 
- Center: Current conversation title (editable inline — click to rename)
- Right:
  * Model selector dropdown (Gemini 1.5 Flash ⚡ / Gemini 1.5 Pro 🧠)
  * Export button (opens Export Chat modal)
  * Share button (opens Share Chat modal)
  * User avatar button → dropdown menu with: Profile, Settings, Keyboard Shortcuts, About, Sign Out

**MODEL SELECTOR DROPDOWN:**
- Floating dropdown card (glassmorphism)
- Option 1: "Gemini 1.5 Flash ⚡" — "Faster responses, great for most tasks"
- Option 2: "Gemini 1.5 Pro 🧠" — "More powerful, better for complex reasoning"
- Selected model shown with checkmark + colored badge
- Changes saved to user preferences in Firebase RTDB

**CONVERSATION ITEM:**
- Shows: title, relative time (e.g. "2h ago", "Yesterday")
- Context menu on right-click OR three-dot button on hover:
  * Rename: inline edit with input
  * Pin/Unpin: toggles isPinned in RTDB
  * Delete: opens Delete Conversation modal (confirmation)
- Optimistic UI: update locally first, then sync to Firebase

**FIREBASE RTDB INTEGRATION:**
- Listen to /conversations/$uid in real-time (onValue)
- Sort by updatedAt descending
- Unsubscribe listener on component unmount
- Handle loading, error, empty states

**SIDEBAR SEARCH:**
- Filters conversation list by title (case-insensitive)
- Shows count: "3 results"
- Clear button (X) when query is not empty
- Empty search state with magnifying glass icon

**RESPONSIVE BEHAVIOR:**
- < 768px: sidebar hidden by default, overlay drawer on open
- ≥ 768px: sidebar always visible, can be collapsed to icon-only (64px) mode
- Collapse button on sidebar edge
- Smooth transition for all states

**LOADING STATES:**
- Full page loader: VortexFlow logo pulsing animation while auth initializes
- Skeleton loaders for conversation list
- Shimmer effect CSS animation

Show ALL files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 5 — Gemini AI Chat Engine & Message System
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 5: Gemini Chat Engine & Message System.

Build the complete real AI chat functionality with Google Gemini streaming API, message storage in Firebase RTDB, and the full chat interface.

CREATE:

**`src/lib/gemini.ts` — COMPLETE IMPLEMENTATION:**
```typescript
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(import.meta.env.VITE_GEMINI_API_KEY);

// System prompt for VortexFlow AI personality
const SYSTEM_PROMPT = `You are VortexFlow AI...` // (full prompt)

// Stream chat response
export async function streamChatResponse(
  messages: Message[],
  model: string,
  onChunk: (chunk: string) => void,
  onComplete: (fullText: string) => void,
  onError: (error: Error) => void
): Promise<void>

// Generate conversation title from first message
export async function generateConversationTitle(firstMessage: string): Promise<string>

// Count approximate tokens
export function estimateTokens(text: string): number
```

**Components in `src/components/chat/`:**
- `ChatWindow.tsx` — main chat area
- `MessageList.tsx` — scrollable list of messages
- `MessageBubble.tsx` — individual message with full markdown
- `MessageActions.tsx` — copy, regenerate, thumbs up/down on hover
- `StreamingMessage.tsx` — animated streaming text display
- `TypingIndicator.tsx` — "VortexFlow is thinking..." with dots animation
- `ChatInput.tsx` — message input bar
- `InputToolbar.tsx` — buttons below input
- `ChatWelcome.tsx` — empty state with prompt suggestions
- `CodeBlock.tsx` — syntax highlighted code with copy button
- `MessageTimestamp.tsx` — hover-reveal timestamp

**CHAT CONTEXT (`src/store/ChatContext.tsx`) — FULL IMPLEMENTATION:**
```typescript
// State:
- conversations: Record<string, Conversation>
- activeConversationId: string | null
- messages: Record<string, Message[]>  // keyed by conversationId
- isStreaming: boolean
- streamingContent: string  // current partial stream
- isLoadingHistory: boolean
- error: string | null

// Actions:
- createNewConversation()
- selectConversation(id)
- sendMessage(content: string)  // main send function
- regenerateLastMessage()
- deleteConversation(id)
- renameConversation(id, title)
- togglePinConversation(id)
- clearAllConversations()
- loadConversationMessages(id)
```

**`sendMessage` FUNCTION FLOW:**
1. Create optimistic user message object → add to local state immediately
2. Write user message to Firebase RTDB `/conversations/$uid/$convId/messages/$msgId`
3. Update conversation `updatedAt` and `messageCount` in RTDB
4. If new conversation: generate title from message using Gemini (async, update after)
5. Start Gemini streaming call with full message history for context
6. As chunks arrive: update `streamingContent` state in real-time
7. On complete: write final assistant message to RTDB
8. Update user stats: increment totalMessages in `/users/$uid/stats`
9. On error: show error message bubble, allow retry

**MESSAGE BUBBLE:**
- User messages: right-aligned, gradient purple background, rounded pill shape
- Assistant messages: left-aligned, dark card bg, with VortexFlow ⚡ avatar
- Markdown rendering with `react-markdown` + `remark-gfm`:
  * Headers, bold, italic, strikethrough
  * Ordered/unordered lists
  * Tables (responsive, styled)
  * Blockquotes
  * Inline code
  * Code blocks with `CodeBlock.tsx`
  * Links (open in new tab)
- Message hover reveals action bar (copy, thumbs up, thumbs down, share)
- Error messages: red tinted card with retry button
- Timestamps shown on hover

**CODE BLOCK COMPONENT:**
- Language badge top-right (e.g. "python", "typescript")
- Copy button top-right → toast "Copied!"
- Syntax highlighting via `highlight.js` with custom dark theme
- Line numbers (optional, toggle)
- Max height 400px with scroll for long code
- Filename support (parse from markdown meta)

**STREAMING MESSAGE:**
- Shows character-by-character appear animation (CSS, not JS loop)
- Blinking cursor at end while streaming
- Smooth scroll to bottom as content grows
- Can't copy/interact while streaming

**CHAT INPUT:**
- Multiline textarea (auto-resize up to 200px)
- Placeholder: "Message VortexFlow AI..."
- Send on Enter (Shift+Enter for newline)
- Send button (disabled when empty or streaming)
- Character counter (show when > 500 chars, warn at 3000, limit at 4000)
- Paste support
- Toolbar below input:
  * Attach file button (UI only, shows "coming soon" tooltip)
  * Voice input button (UI only, shows "coming soon" tooltip)  
  * Model indicator badge (clickable → model selector)
  * "Gemini" powered by badge

**CHAT WELCOME SCREEN (empty state):**
- Large ⚡ logo with glow animation
- "What can I help you with?" heading
- 6 prompt suggestion cards in 2x3 grid:
  1. "Explain quantum computing in simple terms"
  2. "Write a Python script to sort a list"  
  3. "Give me a recipe using chicken and pasta"
  4. "Help me write a professional email"
  5. "What are the best practices for React?"
  6. "Explain the difference between TCP and UDP"
- Click any card → populate input with that prompt
- Fade-in stagger animation

**SCROLL BEHAVIOR:**
- Auto-scroll to bottom on new messages
- Show "scroll to bottom" floating button when user has scrolled up
- Smooth scroll animation
- Don't auto-scroll if user is reading old messages (detect with scroll position)

**FIREBASE RTDB REAL-TIME:**
- Listen to messages for active conversation in real-time
- Use Firebase `onChildAdded` for efficient message loading
- Paginate older messages: load last 50, "Load more" button for older
- Unsubscribe on conversation change

Show ALL files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 6 — All Modals System
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 6: Complete Modals System.

Build all application modals with a professional modal infrastructure and animations.

CREATE:

**Modal Infrastructure:**
- `src/components/modals/ModalPortal.tsx` — React Portal to document.body
- `src/components/modals/ModalOverlay.tsx` — animated backdrop with blur
- `src/components/modals/BaseModal.tsx` — base modal wrapper with animations
- `src/components/modals/ModalManager.tsx` — renders correct modal based on UIContext
- `src/components/ui/Modal.tsx` — reusable modal primitive

**MODAL TYPES ENUM (`src/types/index.ts` addition):**
```typescript
export enum ModalType {
  SETTINGS = "settings",
  PROFILE = "profile",
  KEYBOARD_SHORTCUTS = "keyboard_shortcuts",
  SHARE_CHAT = "share_chat",
  DELETE_CONVERSATION = "delete_conversation",
  CLEAR_ALL_CHATS = "clear_all_chats",
  EXPORT_CHAT = "export_chat",
  FEEDBACK = "feedback",
  ABOUT = "about",
  NEW_CHAT_CONFIRM = "new_chat_confirm",
  RENAME_CONVERSATION = "rename_conversation",
}
```

**1. SETTINGS MODAL (`SettingsModal.tsx`):**
Tabbed modal with 4 tabs:
- **Appearance Tab:**
  * Theme selector: Dark / Light / System (3 visual cards with preview)
  * Font size: Small / Medium / Large (preview text updates live)
  * Sidebar default state: Open / Closed
  * Code theme: Dark / Light
- **AI & Chat Tab:**
  * Model selector with descriptions (Gemini Flash vs Pro)
  * Streaming toggle (on/off with description)
  * System prompt viewer (show current, not editable in free tier)
  * Max context messages slider (5-50, default 20)
- **Notifications Tab:**
  * Sound effects toggle
  * Browser notifications toggle (request permission if enabled)
- **Data & Privacy Tab:**
  * "Export All Data" button → downloads JSON of all conversations
  * "Clear All Chats" button → opens Clear All Chats modal
  * "Delete Account" button (red, destructive) → confirmation
  * Data storage info text
All settings saved to Firebase RTDB `/users/$uid/preferences` in real-time

**2. PROFILE MODAL (`ProfileModal.tsx`):**
- User avatar: large circle with initials/photo + edit button overlay
- Avatar upload: click to select image (convert to base64, store in RTDB — max 200KB)
- Fields:
  * Display Name (editable, with character count 2-50)
  * Email (read-only, shown with lock icon)
  * Bio (textarea, max 200 chars, optional)
- Auth provider badge: "Email/Password" or "Google Account" 
- Account stats:
  * Member since: formatted date
  * Total conversations
  * Total messages sent
- Save button + Cancel button
- Dirty state detection: Save button only enabled when changes made
- Optimistic update + Firebase sync

**3. KEYBOARD SHORTCUTS MODAL (`KeyboardShortcutsModal.tsx`):**
- Clean table layout with two columns: Action | Shortcut
- Sections:
  * Navigation: New Chat (Ctrl+K), Toggle Sidebar (Ctrl+B), Search Chats (Ctrl+/)
  * Chat: Send Message (Enter), New Line (Shift+Enter), Regenerate (Ctrl+R)
  * App: Settings (Ctrl+,), Close Modal (Esc)
- Keyboard key badges styled like real keys (border, shadow, monospace)
- Search/filter shortcuts input
- "macOS" / "Windows" toggle (shows ⌘ vs Ctrl)

**4. SHARE CHAT MODAL (`ShareChatModal.tsx`):**
- Current conversation share options:
  * Copy shareable link (generates a pseudo-link like `vortexflow.ai/share/[id]` — just copies to clipboard, not actually functional)
  * Copy conversation as text
  * Copy as markdown
- Preview pane showing what will be copied
- Toast on copy: "Copied to clipboard!"
- Info note: "Links are view-only snapshots"

**5. DELETE CONVERSATION MODAL (`DeleteConversationModal.tsx`):**
- Shows conversation title in warning message
- Warning icon (triangle)
- Message: "Are you sure you want to delete '[title]'? This action cannot be undone."
- Two buttons: Cancel (ghost) + Delete (red destructive)
- Deletes from Firebase RTDB and local state
- Redirects to most recent conversation or empty state

**6. CLEAR ALL CHATS MODAL (`ClearAllChatsModal.tsx`):**
- Destructive confirmation modal
- Requires typing "DELETE" in input to confirm
- Progress bar filling as user types
- Big warning: "This will permanently delete all X conversations"
- Confirm button only enabled when input matches "DELETE"
- Shows spinner while clearing
- Success toast + redirects to new empty state

**7. EXPORT CHAT MODAL (`ExportChatModal.tsx`):**
- Export current conversation
- Format options as cards:
  * JSON — "Full data with metadata" 
  * Markdown — "Formatted for reading"
  * Plain Text — "Simple text file"
- Preview pane (small, scrollable) showing first ~10 lines
- Filename input (pre-filled with conversation title + date)
- Download button — triggers actual file download
- Export logic from `src/utils/exportChat.ts`

**8. FEEDBACK MODAL (`FeedbackModal.tsx`):**
- "Help us improve VortexFlow AI"
- Type selector: Bug Report / Feature Request / General Feedback
- Rating: 1-5 stars (animated hover)
- Message textarea (min 20 chars)
- Email field (pre-filled from auth, optional)
- Submit → write to Firebase RTDB `/feedback/$pushId`
- Thank you animation on submit

**9. ABOUT MODAL (`AboutModal.tsx`):**
- VortexFlow AI logo + version (v1.0.0)
- "Built with" technology badges: React, TypeScript, Firebase, Gemini API, Tailwind
- Short mission statement
- Links: GitHub (placeholder), Documentation (placeholder)
- Open source badge
- "Made with ❤️ and ⚡" tagline

**10. RENAME CONVERSATION MODAL (`RenameConversationModal.tsx`):**
- Simple modal: current title pre-filled in input
- Character count (max 100)
- Save / Cancel buttons
- Auto-select text on open
- Save on Enter

**BASE MODAL REQUIREMENTS:**
- Framer Motion: scale+fade in, scale+fade out
- Click outside to close
- Escape key to close
- Trap focus within modal (accessibility)
- Prevent body scroll when open
- Max-width varies by modal: sm (400px), md (560px), lg (720px)
- Dark glassmorphism style: bg-[#1A1A26]/95 backdrop-blur-xl border border-white/10
- Header with title + X close button
- Scrollable body for long content
- Footer with action buttons

Show ALL modal files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 7 — UI Component Library
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 7: Reusable UI Component Library.

Build all shared UI components used throughout VortexFlow AI.

CREATE ALL components in `src/components/ui/`:

**1. Button.tsx**
- Variants: primary (gradient), secondary (ghost), outline, destructive, ghost, link
- Sizes: xs, sm, md, lg, xl
- States: loading (spinner), disabled, active
- Icon support: left icon, right icon, icon-only
- Full TypeScript props extending HTMLButtonElement
- Ripple effect animation on click

**2. Input.tsx**
- Variants: default, filled, outline
- States: error, success, disabled, loading
- Left/right icon slots
- Label, helper text, error message
- Character count support
- Password toggle built-in option
- Floating label animation

**3. Textarea.tsx**
- Auto-resize (min/max rows)
- Character count
- Same styling system as Input

**4. Badge.tsx**
- Variants: default, primary, success, warning, error, outline
- Sizes: xs, sm, md
- Dot indicator option
- Removable (X button)

**5. Avatar.tsx**
- Image with fallback to initials
- Sizes: xs(24px), sm(32px), md(40px), lg(56px), xl(80px)
- Status indicator dot (online, away, offline)
- Ring variants

**6. Tooltip.tsx**
- Hover/focus triggered
- Positions: top, bottom, left, right, auto
- Delay option (default 500ms)
- Max width option
- Arrow pointer
- Portal-based rendering

**7. Dropdown.tsx**
- Trigger + content pattern
- Positions: bottom-start, bottom-end, top-start, top-end
- Click outside to close
- Keyboard navigation (arrow keys, Enter, Esc)
- Item variants: default, danger, disabled, separator, label
- Icon support per item
- Animation: scale + fade

**8. Switch.tsx**
- Animated toggle
- Sizes: sm, md, lg
- Label + description option
- Controlled + uncontrolled modes

**9. Slider.tsx**
- Range slider with custom thumb
- Value display
- Min/max/step
- Marks/ticks option

**10. Tabs.tsx**
- Underline, pill, card variants
- Animated active indicator
- Keyboard navigation
- Lazy rendering of tab content

**11. Skeleton.tsx**
- Line skeleton
- Circle skeleton
- Card skeleton
- Pulse animation
- Shimmer animation variant

**12. Spinner.tsx**
- Sizes: xs, sm, md, lg
- Colors: inherit, primary, white
- Pulse variant

**13. Toast System (`src/components/ui/Toast.tsx` + `src/lib/toast.ts`):**
- Types: success, error, warning, info, loading
- Position: top-right (default), configurable
- Auto-dismiss (3000ms default, configurable)
- Manual dismiss X button
- Progress bar showing time remaining
- Stacking with limit (max 4 visible)
- API: `toast.success("msg")`, `toast.error("msg")`, `toast.loading("msg")`, `toast.dismiss(id)`
- Pause on hover
- Entrance/exit animations (slide in from right)

**14. EmptyState.tsx**
- Icon/illustration slot
- Title + description
- Action button slot
- Various preset empties: no-chats, no-results, error-state

**15. Divider.tsx**
- Horizontal/vertical
- With optional label in center

**16. Kbd.tsx** (keyboard key display)
- Styled as physical key
- Single key or combination

**17. CopyButton.tsx**
- Copy text to clipboard
- Animated check state after copy
- Tooltip "Copy" → "Copied!"

**18. ScrollArea.tsx**
- Custom scrollbar styling
- Overflow auto with padding compensation
- Ref forwarding for scroll control

**19. ContextMenu.tsx**
- Right-click triggered
- Portal-based
- Same item API as Dropdown
- Position aware (stays in viewport)

**20. Popover.tsx**
- Click triggered floating content
- Arrow
- Close on outside click
- Used for user menu, model selector

**DESIGN SYSTEM TOKENS in component props:**
All components use the brand color system:
- Primary: violet/purple gradient
- Success: emerald
- Warning: amber  
- Error: red
- Info: blue

Show ALL component files with full implementation. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 8 — Theme System, Keyboard Shortcuts & Accessibility
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 8: Theme System, Keyboard Shortcuts & Accessibility.

BUILD:

**1. Complete Theme System (`src/store/ThemeContext.tsx` + `src/hooks/useTheme.ts`):**
- Themes: "dark" | "light" | "system"
- System theme: uses `prefers-color-scheme` media query + listener
- Theme class applied to `<html>` element: "dark" or "light"
- Persisted in:
  * Firebase RTDB `/users/$uid/preferences/theme` (for authenticated users)
  * localStorage `vortexflow-theme` (fallback/unauthenticated)
- Smooth transition between themes: `transition-colors duration-300`
- No flash on initial load: inline script in index.html to set class before React mounts
- All Tailwind classes use dark: prefix properly

**Light Theme CSS Variables (`src/index.css`):**
```css
:root {
  --bg-primary: #F8F7FF;
  --bg-surface: #FFFFFF;
  --bg-card: #F0EEFF;
  --border: #E2DEFF;
  --text-primary: #1A1530;
  --text-secondary: #6B5FA0;
  --text-muted: #9B97B0;
}
.dark {
  --bg-primary: #0A0A0F;
  --bg-surface: #111118;
  --bg-card: #1A1A26;
  --border: #2A2A3A;
  --text-primary: #F0EDFF;
  --text-secondary: #C4BFDF;
  --text-muted: #6B6880;
}
```

**2. Keyboard Shortcuts System (`src/hooks/useKeyboard.ts`):**
Global keyboard shortcut handler:
- `Ctrl+K` / `Cmd+K` → New Chat
- `Ctrl+B` / `Cmd+B` → Toggle Sidebar
- `Ctrl+/` / `Cmd+/` → Focus search
- `Ctrl+,` / `Cmd+,` → Open Settings modal
- `Ctrl+R` / `Cmd+R` → Regenerate last message
- `Ctrl+Shift+C` → Copy last message
- `Escape` → Close modal / clear search
- `Ctrl+Enter` → Force send message
- `Up Arrow` in empty input → Edit last user message
- Register/unregister system so components can add their own shortcuts
- Shortcut blocked when: typing in input (except app-level), modal open (only Esc works)
- Visual feedback: flash effect on triggered action

**3. Font Size System:**
- Small: base text 13px, code 12px
- Medium: base text 15px, code 13px (default)
- Large: base text 17px, code 15px
- Applied via class on root element: `text-size-sm`, `text-size-md`, `text-size-lg`
- CSS custom property `--font-size-base` used in components
- Saved to Firebase preferences

**4. Accessibility (a11y) Improvements:**
- All interactive elements have proper `aria-label`
- Focus visible styles on all focusable elements (3px purple outline)
- Skip to main content link (visible on Tab press)
- Proper heading hierarchy (h1 > h2 > h3)
- Color contrast ratios pass WCAG AA
- All modals: role="dialog", aria-modal, aria-labelledby, aria-describedby
- Toast notifications: aria-live="polite" region
- Loading states: aria-busy, aria-label="Loading..."
- Image alt texts on all images/avatars
- Keyboard navigation for: sidebar list (arrow keys), dropdown menus, tab panels
- Screen reader announcements for: new message received, streaming complete, errors

**5. Smooth Page Transitions (`src/components/routing/PageTransition.tsx`):**
- Wrap page components with Framer Motion
- Route change: fade out old page, fade in new page
- No layout shift during transition

**6. Micro-interactions & Polish:**
- New chat button: hover sparkle particle burst (CSS only)
- Send button: elastic scale on click
- Message bubble: appears with slide-up animation
- Sidebar item: hover slides right 2px
- Copy button: success state with bouncing checkmark
- Toggle switches: smooth slide with spring physics
- Modal backdrop: blur and dim animate in sync

**7. Error Boundaries (`src/components/error/`):**
- `ErrorBoundary.tsx` — catches render errors, shows friendly error UI
- `ChatErrorBoundary.tsx` — chat-specific error with retry
- `RouteErrorBoundary.tsx` — route-level error with navigation back
- Error UI: shows "Something went wrong" with ⚡ broken logo, refresh button

**8. Performance Optimizations:**
- All page components: `React.lazy` + `Suspense` with skeleton loaders
- Heavy components memoized with `React.memo`
- Expensive computations wrapped in `useMemo`
- Event handlers in `useCallback`
- Images lazy loaded
- Firebase listeners properly cleaned up
- Debounced search (300ms)
- Throttled scroll handlers (16ms)

**9. 404 Page (`src/pages/NotFoundPage.tsx`):**
- Large "404" with glitch animation
- "Page not found" message
- "The page you're looking for doesn't exist in this dimension."
- "Go Home" and "Open Chat" buttons
- Animated floating ⚡ particles background

Show ALL files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 9 — Chat Features: Search, Export, Regenerate, Edit
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 9: Advanced Chat Features.

BUILD all advanced chat functionality:

**1. Message Search (`src/components/chat/MessageSearch.tsx`):**
- Search within current conversation
- Highlight matching text in messages
- Navigate between results (up/down arrows)
- Result count: "3 of 12 matches"
- Triggered by Ctrl+F within chat window
- Animated slide-down from header
- Close on Escape

**2. Regenerate Response:**
- "Regenerate" button on last assistant message
- Re-sends last user message to Gemini
- Replaces last assistant message (not appends)
- Shows loading state while streaming
- Toast: "Regenerating response..."
- Update in Firebase RTDB

**3. Edit User Message:**
- Click edit icon on user message → inline edit mode
- Textarea replaces message bubble with original content
- Save / Cancel buttons
- On save: deletes all messages after edited message, re-sends to Gemini
- Firebase: delete subsequent messages, update edited message
- Warning: "Editing this message will remove all subsequent messages"

**4. Message Actions Bar (hover):**
Per message on hover:
- User messages: Copy, Edit, Delete
- Assistant messages: Copy, Regenerate (last only), Thumbs Up, Thumbs Down, Share
- Thumbs up/down: saves to RTDB `/conversations/$uid/$convId/messages/$msgId/feedback`
- Delete message: removes from RTDB + local state (with confirmation toast)
- Actions animate in with stagger

**5. Global Chat Search (`src/components/sidebar/GlobalSearch.tsx`):**
- Search across all conversations by title AND message content
- Triggered by Ctrl+/ or clicking search bar
- Full-screen overlay search panel
- Shows conversation results grouped:
  * "Title matches" (search in titles)
  * "Message matches" (search in content, shows excerpt)
- Click result → navigate to conversation, scroll to message
- Recent searches (last 5, stored in localStorage)
- Keyboard navigation through results
- Debounced 300ms

**6. Message Timestamps:**
- Show relative time on hover ("2 minutes ago", "Yesterday at 3:45 PM")
- Tooltip with absolute timestamp
- Group messages by time: show time separator if > 10 min gap between messages

**7. Conversation Auto-Title Generation:**
- When creating new conversation, title = "New Chat"
- After first user message sent + AI responds:
  * Call Gemini with prompt: "Generate a short 3-6 word title for a conversation that starts with: [message]. Return ONLY the title, nothing else."
  * Update conversation title in RTDB
  * Animate title update in sidebar

**8. Message Streaming Polish:**
- Smooth character-by-character render (no jitter)
- Typing cursor blinks at insertion point
- Markdown renders progressively (headings, bullets appear as they stream)
- Code blocks appear when ``` fence is closed
- Scroll follows content smoothly
- Stop streaming button (X) in input area while streaming
  * Cancels fetch, saves partial response to RTDB

**9. Chat Context Window Management:**
- Show token usage indicator in header: "2.4K / 32K tokens"
- When approaching limit: yellow warning "Approaching context limit"
- At limit: auto-summarize oldest messages or show warning
- Settings: max context messages slider (5-50)
- Trim old messages from Gemini API call (not from RTDB, just from context sent)

**10. Quick Actions in Chat:**
- `/` command menu in input:
  * Type "/" → shows floating menu above input
  * Options: /new (new chat), /clear (clear current), /export (export chat), /model (switch model)
  * Arrow key navigation, Enter to select, Esc to close
  * Animated slide-up menu

**11. Pinned Messages:**
- Pin any message (right-click → pin)
- Pinned messages panel: collapsible section at top of chat
- Store in RTDB: `/conversations/$uid/$convId/messages/$msgId/isPinned: true`
- Click pinned message → scroll to it

**12. Copy Code & Share Improvements:**
- Copy entire conversation as formatted markdown
- Copy single message
- All copy operations use Clipboard API with fallback
- Success feedback: button turns green with checkmark for 2s

Show ALL files completely. Do not truncate.
```

---

# ═══════════════════════════════════════════════
# MILESTONE 10 — Final Polish, Animations & Production Build
# ═══════════════════════════════════════════════

### 🔨 PASTE THIS INTO AI STUDIO:

```
Continuing VortexFlow AI build. This is Milestone 10: Final Polish, Animations & Production Readiness.

BUILD the final layer of polish and complete all remaining pieces:

**1. Loading Experience (`src/components/ui/AppLoader.tsx`):**
- Full-screen loader shown while Firebase initializes auth
- Animated VortexFlow ⚡ logo with:
  * Rotating outer ring (gradient arc)
  * Pulsing center icon
  * "VortexFlow AI" text fades in below
  * Subtitle: "Initializing..." → "Loading your workspace..." → "Almost ready..."
- Progress indicator
- Minimum display time: 1.5s (avoid flash)

**2. Onboarding Flow (`src/components/onboarding/`):**
First-time user experience after signup:
- `OnboardingModal.tsx` — multi-step modal (3 steps):
  * Step 1: Welcome + name display, "Let's set up your experience"
  * Step 2: Theme selection (Dark/Light) with live preview
  * Step 3: Model selection (Flash vs Pro) with description
- Progress dots at bottom
- Skip option
- Finish → writes preferences to RTDB, shows welcome toast
- Never shown again (track in RTDB `/users/$uid/onboardingComplete: true`)

**3. Network Status Banner (`src/components/ui/NetworkStatus.tsx`):**
- Listen to `online`/`offline` events
- When offline: yellow banner at top "⚠️ You're offline. Messages will send when reconnected."
- When reconnects: green banner "✅ Back online!" → auto-dismiss after 3s
- Firebase RTDB automatically retries when back online
- Slide-down animation

**4. User Menu Dropdown (complete implementation):**
Clicking user avatar in header shows popover:
- User info: avatar, name, email
- Divider
- Menu items with icons:
  * 👤 Profile (opens Profile modal)
  * ⚙️ Settings (opens Settings modal)
  * ⌨️ Keyboard Shortcuts (opens Shortcuts modal)
  * 📤 Export Data (downloads all data as JSON)
  * 💬 Send Feedback (opens Feedback modal)
  * ℹ️ About VortexFlow (opens About modal)
- Divider
- 🚪 Sign Out (with confirmation)
- Smooth Framer Motion animation

**5. Empty States Polish:**
- New user first visit: animated hero in chat area
  * "Welcome to VortexFlow AI, [Name]! 👋"
  * 3 starter prompts as animated cards
- No conversations in sidebar: illustration + "Start your first conversation"
- All empty states have consistent spacing, icon size, typography

**6. Error Handling Polish:**
- Firebase connection errors: specific messages
- Gemini API errors:
  * Rate limit: "You've sent too many messages. Please wait a moment."
  * Network error: "Couldn't reach AI. Check your connection."
  * Content filter: "This message was flagged. Please rephrase."
  * Generic: "Something went wrong. Please try again."
- Auth errors mapped to friendly messages:
  * wrong-password → "Incorrect password. Please try again."
  * user-not-found → "No account found with this email."
  * email-already-in-use → "An account with this email already exists."
  * network-request-failed → "Network error. Check your connection."

**7. Responsive Chat Layout Polish:**
- Mobile (< 640px):
  * Sidebar hidden, hamburger menu
  * Full-width messages
  * Input bar fixed to bottom with safe area inset
  * Header compact (just toggle + title + user avatar)
  * Touch-friendly tap targets (min 44x44px)
- Tablet (640px-1024px):
  * Sidebar collapsible
  * Slightly narrower message bubbles
- Desktop (> 1024px):
  * Full sidebar
  * Max message width 70% for readability

**8. Performance Audit & Fixes:**
- Bundle size optimization: check for duplicate dependencies
- Code splitting: each page/modal is its own chunk
- Firebase listeners audit: verify all unsubscribed on unmount
- Memory leak check: all timers/intervals cleared
- Image optimization: avatar stored as base64 compressed

**9. Environment Variables & Config:**
Create `.env.example`:
```env
VITE_FIREBASE_API_KEY=
VITE_FIREBASE_AUTH_DOMAIN=
VITE_FIREBASE_DATABASE_URL=
VITE_FIREBASE_PROJECT_ID=
VITE_FIREBASE_STORAGE_BUCKET=
VITE_FIREBASE_MESSAGING_SENDER_ID=
VITE_FIREBASE_APP_ID=
VITE_GEMINI_API_KEY=
```
Create `src/config/env.ts` — typed env variable access with validation

**10. README.md:**
Complete setup guide:
- Project overview with screenshot placeholder
- Features list
- Prerequisites
- Firebase setup steps (create project, enable Auth methods, RTDB rules)
- Getting API keys (Firebase + Gemini)
- Installation steps
- Environment variable setup
- Running locally
- Build for production
- Firebase Hosting deployment steps
- RTDB Security Rules (provided, paste into Firebase Console):
```json
{
  "rules": {
    "users": {
      "$uid": {
        ".read": "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    },
    "conversations": {
      "$uid": {
        ".read": "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    },
    "feedback": {
      ".write": "auth !== null",
      ".read": false
    }
  }
}
```

**11. Final App.tsx — Complete Router:**
```typescript
// All routes:
/ → LandingPage (PublicRoute)
/login → LoginPage (PublicRoute)
/signup → SignupPage (PublicRoute)
/forgot-password → ForgotPasswordPage (PublicRoute)
/chat → ChatPage (ProtectedRoute)
/chat/:conversationId → ChatPage (ProtectedRoute)
* → NotFoundPage
```

**12. PWA Manifest (`public/manifest.json`):**
- App name, short name, icons, theme colors
- Display: standalone
- Start URL: /chat
- Background color matching brand

**13. Meta Tags & SEO (`index.html` final):**
- Title: "VortexFlow AI — Your Professional AI Assistant"
- Description meta
- Open Graph tags (og:title, og:description, og:image placeholder)
- Twitter Card tags
- Theme color meta
- Viewport meta (with viewport-fit=cover for iPhone notch)

**14. Final Touches:**
- Favicon: SVG ⚡ in brand purple
- Console.log suppressed in production
- Global error handler: uncaught promise rejections logged
- Version display in About modal and footer: `v1.0.0`
- Smooth page title updates: "[Chat Title] — VortexFlow AI"

Show ALL files completely. Final milestone — ensure everything integrates perfectly. Do not truncate.
```

---

## 📋 POST-BUILD CHECKLIST

After completing all milestones, verify:

- [ ] Firebase Auth works (email + Google)
- [ ] New user created in RTDB on signup
- [ ] Conversations save and load in real-time
- [ ] Gemini streaming works with correct API key
- [ ] All 10+ modals open/close correctly
- [ ] Theme switching works (dark/light/system)
- [ ] All keyboard shortcuts functional
- [ ] Export chat downloads file correctly
- [ ] Mobile responsive on all breakpoints
- [ ] No console errors in production build
- [ ] `npm run build` completes without TypeScript errors
- [ ] Firebase RTDB security rules deployed

---

## 🚀 DEPLOY TO FIREBASE HOSTING

```bash
npm install -g firebase-tools
firebase login
firebase init hosting
npm run build
firebase deploy
```

---

*VortexFlow AI — Built with React + TypeScript + Vite + Firebase + Google Gemini*  
*Roadmap Version: 1.0 | Total Milestones: 10 | Estimated Lines: 8000+*
