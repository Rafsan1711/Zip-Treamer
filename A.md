# VortexFlow AI — Master Roadmap & Prompt Guide
### For Google AI Studio (Gemini) → React + TypeScript + Vite

> **How to use:** Each Milestone = One response from AI Studio Build.
> Paste the relevant Milestone section as your prompt. The AI will build that section completely before you move to the next.
> **Stack:** React 19 + TypeScript + Vite + Firebase Auth + Firebase RTDB + Tailwind CSS + Framer Motion + Google Gemini API

---

## 🎨 Design System (Global — Read Every Milestone)

```
PRIMARY COLOUR PALETTE (Dark Mode Only):
  Background:       #0A0A0F  (near-black, deep space)
  Surface:          #111118  (card/panel bg)
  Surface-2:        #1A1A24  (elevated surface)
  Surface-3:        #22222E  (hover states)
  Border:           #2A2A3A  (subtle dividers)
  Border-bright:    #3D3D52  (active borders)

ACCENT COLOURS:
  Primary:          #00D4FF  (electric cyan — main brand)
  Primary-glow:     rgba(0,212,255,0.15)
  Secondary:        #7B61FF  (deep indigo/blue-violet — NOT pink/violet)
  Secondary-glow:   rgba(123,97,255,0.15)
  Accent-warm:      #FF6B35  (ember orange — CTAs, warnings)
  Success:          #00E5A0  (mint green)
  Error:            #FF4D6A  (crimson)
  Warning:          #FFB830  (amber)

TEXT:
  text-primary:     #F0F0FF
  text-secondary:   #9898B8
  text-muted:       #5C5C7A

GRADIENTS:
  Brand gradient:   linear-gradient(135deg, #00D4FF 0%, #7B61FF 100%)
  Hero gradient:    radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.08) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 20%, rgba(123,97,255,0.08) 0%, transparent 60%)
  Card shine:       linear-gradient(135deg, rgba(255,255,255,0.05) 0%, transparent 100%)

FONTS:
  Heading: 'Inter' (700, 800)
  Body:    'Inter' (400, 500)
  Mono:    'JetBrains Mono' (code blocks)
  Load from Google Fonts

ICONS: Font Awesome 6 Pro (use @fortawesome/react-fontawesome + free solid/regular/brands sets)

ANIMATION PRINCIPLES:
  - Framer Motion for all page transitions (opacity + y: 20 → 0, duration 0.4)
  - CSS cubic-bezier(0.16, 1, 0.32, 1) for all hovers
  - Glassmorphism: backdrop-filter: blur(12px), bg rgba(255,255,255,0.04)
  - Glow effects on interactive elements on hover
  - No jarring animations, everything smooth and premium
```

---

## 📁 Project Structure (Final Vision)

```
vortexflow-ai/
├── public/
│   └── logo.svg              ← Your AI-generated logo goes here
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── ui/               ← Button, Input, Modal, Badge, Tooltip, etc.
│   │   ├── chat/             ← ChatWindow, MessageBubble, InputBar, etc.
│   │   ├── layout/           ← Sidebar, Navbar, Footer
│   │   └── modals/           ← SettingsModal, ProfileModal, etc.
│   ├── pages/
│   │   ├── LandingPage.tsx
│   │   ├── AuthPage.tsx
│   │   └── ChatPage.tsx
│   ├── hooks/                ← useAuth, useChat, useFirebase, useTheme, etc.
│   ├── lib/
│   │   ├── firebase.ts
│   │   ├── gemini.ts
│   │   └── utils.ts
│   ├── store/                ← Zustand global state
│   ├── types/                ← All TypeScript interfaces
│   ├── styles/               ← global.css, tailwind config
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── tailwind.config.ts
├── vite.config.ts
└── package.json
```

---

---

# ═══════════════════════════════════════
# MILESTONE 1 — Project Foundation & Landing Page
# ═══════════════════════════════════════

## 🎯 PASTE THIS AS YOUR FIRST AI STUDIO PROMPT:

```
You are an expert React + TypeScript + Vite developer. Build "VortexFlow AI" — a professional AI chatbot platform. This is Milestone 1.

TASK: Set up the complete project foundation and build a stunning, fully animated Landing Page.

─────────────────────────────────────────
TECH STACK:
- React 19 + TypeScript + Vite
- Tailwind CSS v3 (configured with custom design tokens)
- Framer Motion (animations)
- React Router DOM v6
- Font Awesome 6 (@fortawesome/react-fontawesome, @fortawesome/free-solid-svg-icons, @fortawesome/free-brands-svg-icons, @fortawesome/free-regular-svg-icons)
- Zustand (state management)
- Firebase v10 (auth + RTDB — config placeholders for now)
- Google Fonts: Inter (400,500,700,800) + JetBrains Mono (400,500)

─────────────────────────────────────────
DESIGN SYSTEM (apply globally):

Background:    #0A0A0F
Surface:       #111118
Surface-2:     #1A1A24
Surface-3:     #22222E
Border:        #2A2A3A
Border-bright: #3D3D52

Primary:       #00D4FF  (electric cyan)
Primary-glow:  rgba(0,212,255,0.15)
Secondary:     #7B61FF  (deep indigo)
Secondary-glow:rgba(123,97,255,0.15)
Accent-warm:   #FF6B35  (ember orange)
Success:       #00E5A0
Error:         #FF4D6A
Warning:       #FFB830

text-primary:  #F0F0FF
text-secondary:#9898B8
text-muted:    #5C5C7A

Brand gradient: linear-gradient(135deg, #00D4FF 0%, #7B61FF 100%)
NO pink, NO violet-pink. Dark mode ONLY. Premium quality like Claude/ChatGPT/Perplexity dark UI.

─────────────────────────────────────────
FILES TO CREATE:

1. package.json — all dependencies listed above
2. vite.config.ts — path aliases (@/ → src/)
3. tailwind.config.ts — full custom config with all design tokens as CSS variables + Tailwind extensions
4. index.html — Google Fonts import, Font Awesome kit, base meta tags for VortexFlow AI
5. src/main.tsx — app entry
6. src/App.tsx — React Router setup with routes: / (Landing), /auth (Auth), /chat (Chat — placeholder for now)
7. src/styles/global.css — CSS variables, scrollbar styling, base resets, glassmorphism utility classes
8. src/types/index.ts — TypeScript interfaces: User, Message, Chat, ChatSettings, Model, etc.
9. src/lib/firebase.ts — Firebase init with ENV variable placeholders (import.meta.env.VITE_FIREBASE_*)
10. src/lib/utils.ts — cn() classname util, formatDate, truncateText, generateId helpers
11. src/store/useAppStore.ts — Zustand store: user, chats, activeChat, settings, ui state (sidebar open, modal states)

12. src/pages/LandingPage.tsx — FULL LANDING PAGE (described below)

─────────────────────────────────────────
LANDING PAGE — FULL SPEC:

The landing page must be world-class, scroll-based animated, premium dark UI. Sections:

[A] NAVBAR (sticky, glassmorphism on scroll):
- Left: logo.svg (src="/logo.svg") with animated glow pulse on hover, "VortexFlow AI" text in brand gradient
- Center: Nav links — Features, Pricing (shows "Free" badge), About, Changelog
- Right: "Sign In" ghost button + "Get Started Free" CTA button (brand gradient bg, orange glow on hover)
- Mobile: hamburger menu with smooth slide-down drawer
- On scroll >50px: backdrop-blur glassmorphism effect activates

[B] HERO SECTION:
- Full viewport height
- Background: deep space dark + animated radial gradient orbs (cyan top-left, indigo bottom-right) — CSS animated, not canvas
- Floating badge: "✦ Powered by Google Gemini" with subtle pulse
- H1: "The AI That Thinks  With You" — "Thinks" word has brand gradient text + subtle glow
- Subheadline: "VortexFlow AI brings you next-generation conversations — intelligent, fast, and beautifully designed. Start for free, no credit card needed."
- Two CTAs: "Start Chatting Free →" (primary, brand gradient) + "See How It Works" (ghost)
- Animated hero mockup below: a fake-but-realistic chat UI preview (3-4 static message bubbles, typewriter effect on last AI message), framed in a sleek dark card with gradient border
- Scroll-down indicator: animated chevron

[C] FEATURES SECTION ("Everything you need, nothing you don't"):
- 6 feature cards in 3x2 grid (responsive)
- Glassmorphism cards with gradient border on hover + icon glow
- Features:
  1. ⚡ Lightning Fast — Sub-second responses powered by Gemini Flash
  2. 🧠 Context Memory — Remembers your full conversation history
  3. 🔒 Private & Secure — Firebase Auth, your data stays yours
  4. 💻 Code Intelligence — Syntax highlighting, copy button, language detection
  5. 🌐 Real-time Sync — Chats sync instantly across all your devices
  6. 🎨 Beautiful UI — Dark mode premium interface, built for focus
- Stagger animation on scroll into view (Framer Motion useInView)

[D] HOW IT WORKS ("Three steps to smarter conversations"):
- Horizontal 3-step flow with connecting animated line
- Step 1: Create Account (icon: user-plus)
- Step 2: Start a Chat (icon: comments)
- Step 3: Get Instant AI Answers (icon: bolt)
- Each step has number badge, icon, title, description

[E] STATS BAR (full-width, dark surface-2):
- 4 animated counters (count up on scroll into view):
  - "10M+" Conversations
  - "99.9%" Uptime
  - "<1s" Response Time
  - "100%" Free Forever*
- Subtle gradient dividers between stats

[F] TESTIMONIALS ("Loved by curious minds"):
- 3 testimonial cards, horizontal scroll on mobile
- Glassmorphism cards, star ratings, avatar placeholder with gradient initials
- Real-looking names, realistic short testimonials about the AI chat experience

[G] CTA BANNER:
- Full-width section, brand gradient background (subtle, not garish)
- "Ready to think bigger?" headline
- "Join thousands already using VortexFlow AI for free"
- Big "Get Started — It's Free →" button

[H] FOOTER:
- Logo + tagline
- Links: Features, Privacy Policy, Terms of Service, Contact
- Social icons (Twitter/X, GitHub, Discord) using Font Awesome brands
- "© 2025 VortexFlow AI. Built with ♥ and Gemini."
- Subtle top border gradient

─────────────────────────────────────────
ANIMATION REQUIREMENTS:
- Framer Motion: stagger children, fade+slide up on scroll (useInView)
- Hero orbs: CSS keyframe animation (float + pulse)
- Navbar: smooth bg transition on scroll (useState + useEffect)
- Feature cards: scale(1.02) + border glow on hover
- CTA buttons: glow pulse on hover, scale on click
- Stat counters: count-up animation when in viewport
- All transitions: cubic-bezier(0.16, 1, 0.32, 1)

─────────────────────────────────────────
IMPORTANT RULES:
- TypeScript strict mode, no `any`
- All components properly typed
- Mobile responsive (Tailwind breakpoints)
- No placeholder "lorem ipsum" — use real VortexFlow AI copy
- No pink/violet-pink colours anywhere
- Dark mode only — no light mode toggle needed yet
- The logo at /logo.svg will be placed by the user — use an <img> tag with fallback
- Leave /auth and /chat as simple placeholder pages for now ("Coming in Milestone 2/3")
- Export everything properly, clean file structure

Build everything now. Output all files completely.
```

---

---

# ═══════════════════════════════════════
# MILESTONE 2 — Authentication System
# ═══════════════════════════════════════

## 🎯 PASTE THIS AS YOUR MILESTONE 2 PROMPT:

```
Continue building "VortexFlow AI". This is Milestone 2. The project foundation and Landing Page from Milestone 1 already exist. Do not rebuild them — only add/modify the specified files.

TASK: Build the complete Authentication System with a beautiful Auth Page.

─────────────────────────────────────────
AUTH PROVIDERS (Firebase):
- Email + Password (with email verification flow)
- Google OAuth (Sign in with Google)

─────────────────────────────────────────
FILES TO CREATE/MODIFY:

1. src/lib/firebase.ts — Complete Firebase init:
   - Firebase Auth (getAuth, GoogleAuthProvider)
   - Firebase RTDB (getDatabase)
   - All using import.meta.env.VITE_FIREBASE_* env vars
   - Export: auth, db, googleProvider

2. src/hooks/useAuth.ts — Custom hook:
   - onAuthStateChanged listener
   - signInWithEmail(email, pass)
   - signUpWithEmail(email, pass, displayName)
   - signInWithGoogle()
   - signOut()
   - sendEmailVerification()
   - resetPassword(email)
   - Returns: { user, loading, error }
   - After sign-in: write user profile to RTDB at users/{uid}

3. src/components/ui/Input.tsx — Reusable input:
   - label, placeholder, type, error, icon (FA), rightIcon props
   - Dark styled, focus ring in primary cyan, error state in red

4. src/components/ui/Button.tsx — Reusable button:
   - variants: primary (brand gradient), secondary (ghost), danger, outline
   - size: sm, md, lg
   - loading state (spinner icon)
   - icon support (left/right)

5. src/pages/AuthPage.tsx — FULL AUTH PAGE:

   Layout: Split screen on desktop (left: branding panel, right: form)
   Mobile: Single column, form only

   LEFT PANEL (desktop only, ~45% width):
   - Brand gradient background (subtle, dark)
   - Animated floating orbs (same as hero)
   - Logo + "VortexFlow AI"
   - Tagline: "Your AI. Your conversations. Your control."
   - 3 feature bullet points with FA icons:
     ✓ Free forever, no credit card
     ✓ Powered by Google Gemini
     ✓ Sync across all devices
   - Bottom: real user testimonial card (glassmorphism)

   RIGHT PANEL (form area):
   - Glassmorphism card, centered
   - Tabs: "Sign In" | "Sign Up" (smooth animated underline indicator)
   
   SIGN IN TAB:
   - "Welcome back" heading
   - Google Sign In button (full width, white bg, Google icon, dark text "Continue with Google")
   - Divider: "── or continue with email ──"
   - Email input (with envelope icon)
   - Password input (with lock icon + show/hide toggle eye icon)
   - "Forgot password?" link (right aligned, opens forgot password view)
   - Sign In button (primary, full width, loading state)
   - Bottom: "Don't have an account? Sign up →"

   SIGN UP TAB:
   - "Create your account" heading
   - Google Sign Up button
   - Divider
   - Full Name input (user icon)
   - Email input
   - Password input + strength indicator bar (weak/fair/strong/very strong with colour)
   - Confirm Password input
   - Terms checkbox: "I agree to Terms of Service and Privacy Policy"
   - Create Account button (primary, full width, loading state)
   - Bottom: "Already have an account? Sign in →"

   FORGOT PASSWORD VIEW (animated slide-in):
   - "Reset your password" heading
   - Email input
   - Send Reset Email button
   - Success state: green checkmark animation + "Check your inbox" message
   - Back to Sign In link

   EMAIL VERIFICATION BANNER (shown after signup before verification):
   - Full-width amber warning banner
   - "Please verify your email. Resend email" link
   - Dismiss button

   ERROR HANDLING:
   - Firebase error codes mapped to human-readable messages
   - Inline error display under inputs
   - Toast notification for success states

6. src/components/ui/Toast.tsx — Toast notification system:
   - Types: success, error, warning, info
   - Auto-dismiss (configurable duration)
   - Stack multiple toasts (bottom-right)
   - FA icons per type
   - Framer Motion slide-in/out

7. src/store/useAppStore.ts — ADD:
   - toasts array + addToast, removeToast actions
   - authLoading state

8. src/App.tsx — MODIFY:
   - Protected route logic: /chat requires auth, redirect to /auth if not
   - If already authed and visiting /auth, redirect to /chat
   - Show full-page loading spinner during auth state check

─────────────────────────────────────────
ANIMATIONS:
- Tab switch: Framer Motion AnimatePresence for form content
- Forgot password: slide-in from right
- Google button: subtle scale on hover + shadow
- Password strength bar: animated width transition
- Form fields: shake animation on error submit
- Success checkmark: stroke draw animation (SVG)

─────────────────────────────────────────
IMPORTANT:
- After successful auth (any method), navigate to /chat
- Store user in Zustand store
- RTDB user node structure:
  users/{uid}: { uid, displayName, email, photoURL, createdAt, lastSeen, plan: "free" }
- No pink/violet-pink. Premium dark UI consistent with Landing Page.
- All TypeScript strict, no `any`.

Build all specified files completely.
```

---

---

# ═══════════════════════════════════════
# MILESTONE 3 — Main Chat Interface (Core)
# ═══════════════════════════════════════

## 🎯 PASTE THIS AS YOUR MILESTONE 3 PROMPT:

```
Continue building "VortexFlow AI". This is Milestone 3. Milestones 1 and 2 are complete (Landing Page + Auth). Do not rebuild them.

TASK: Build the core Chat Interface — the main application shell, sidebar, and chat window.

─────────────────────────────────────────
FILES TO CREATE/MODIFY:

1. src/lib/gemini.ts — Google Gemini API integration:
   - Uses import.meta.env.VITE_GEMINI_API_KEY
   - GoogleGenerativeAI SDK (@google/generative-ai)
   - Model: gemini-2.0-flash (default), also support gemini-1.5-pro
   - sendMessage(messages: Message[], settings: ChatSettings): AsyncGenerator (streaming)
   - Uses streaming (generateContentStream)
   - System prompt support
   - Token counting helper

2. src/hooks/useChat.ts — Chat management hook:
   - createNewChat(title?: string): string (returns chatId)
   - sendMessage(chatId, content): Promise<void>
   - deleteChat(chatId): Promise<void>
   - updateChatTitle(chatId, title): Promise<void>
   - Streaming response handling (token by token)
   - All data persisted to Firebase RTDB:
     chats/{uid}/{chatId}: { id, title, createdAt, updatedAt, messageCount, model }
     messages/{uid}/{chatId}/{msgId}: { id, role, content, timestamp, tokens }
   - Auto-generate chat title from first message (first 40 chars)
   - Real-time listener for chat list (onValue)

3. src/components/layout/Sidebar.tsx — Left sidebar:
   WIDTH: 260px (collapsible to 0 on mobile, 60px icon-only mode on desktop toggle)
   
   TOP SECTION:
   - Logo + "VortexFlow AI" (hidden in icon-only mode)
   - "New Chat" button (brand gradient, FA plus icon, full width)
   - Collapse toggle button (FA chevron icon)
   
   MIDDLE SECTION (scrollable chat list):
   - Search bar to filter chats (FA magnifying glass icon)
   - Chat groups by date: "Today", "Yesterday", "Previous 7 Days", "Older"
   - Each chat item:
     - Title (truncated, 1 line)
     - Timestamp (relative: "2h ago")
     - Hover: show edit (pencil) + delete (trash) icon buttons
     - Active state: brand gradient left border + bg highlight
     - Click: navigate/open that chat
   - Empty state: "No chats yet. Start a new conversation!"
   - Smooth skeleton loading state while chats load

   BOTTOM SECTION:
   - User avatar (from Google photoURL or gradient initials fallback)
   - User name + email (truncated)
   - Plan badge: "Free" (amber outline)
   - Settings gear icon button (opens Settings Modal)
   - Sign out button (FA arrow-right-from-bracket icon)

4. src/components/layout/ChatLayout.tsx — Main layout wrapper:
   - Flex row: Sidebar + Main area
   - Mobile overlay sidebar (backdrop blur overlay when open)
   - Responsive breakpoints

5. src/pages/ChatPage.tsx — Chat page router:
   - URL: /chat (new chat) + /chat/:chatId (specific chat)
   - Renders ChatLayout with correct chatId
   - If no chatId, show welcome screen
   - React Router useParams for chatId

6. src/components/chat/WelcomeScreen.tsx — Shown when no chat is active:
   - Center of screen
   - Logo (animated glow)
   - "How can I help you today?" heading
   - 4 suggestion cards in 2x2 grid:
     - "Explain quantum computing simply"
     - "Write a Python web scraper"
     - "Help me plan my week"
     - "What's the history of the internet?"
   - Clicking a suggestion starts a new chat with that as the first message
   - Each card: glassmorphism, FA icon, hover glow

7. src/components/chat/ChatWindow.tsx — Main chat display:
   - Scrollable message list (auto-scroll to bottom on new message)
   - Message grouping (consecutive same-role messages)
   - Auto-scroll with smart behavior (don't force scroll if user scrolled up)
   - "Scroll to bottom" FAB button (appears when scrolled up)
   - Empty chat: show minimal centered prompt hint

8. src/components/chat/MessageBubble.tsx — Individual message:
   USER MESSAGES:
   - Right-aligned (or left with distinct bg)
   - Surface-2 background, rounded corners
   - Edit button on hover (pencil FA icon)
   - Copy button on hover

   AI MESSAGES:
   - Left-aligned, no background bubble (clean, like Claude/Perplexity)
   - VortexFlow logo/avatar icon left
   - Markdown rendering (react-markdown + remark-gfm)
   - Code blocks: syntax highlighted (react-syntax-highlighter, Dracula theme or dark theme)
   - Code block header: language label + copy button
   - Inline code: styled monospace
   - Tables: styled dark theme
   - Lists, blockquotes: properly styled
   - Streaming: show text as it arrives, blinking cursor at end
   - Action bar below (appears on hover): Copy, Regenerate, Like 👍, Dislike 👎
   - Token count (small, muted, below message)

   SYSTEM MESSAGES (errors, info):
   - Centered, amber/red styling

9. src/components/chat/InputBar.tsx — Message input area:
   - Sticky at bottom
   - Glassmorphism background
   - Textarea (auto-resize, max 200px height, min 1 line)
   - Placeholder: "Message VortexFlow AI..."
   - Left icons: FA paperclip (disabled, "coming soon" tooltip), FA microphone (disabled)
   - Right: Character/token counter (shows when >100 chars) + Send button
   - Send button: brand gradient, FA paper-plane icon, disabled when empty/loading
   - Enter = send, Shift+Enter = new line
   - Loading state: pulsing stop button (FA square icon) to cancel streaming
   - Below input: "VortexFlow AI can make mistakes. Verify important info." in muted text
   - Model selector badge (shows current model, clickable — opens model picker)

10. src/components/ui/TypingIndicator.tsx:
    - 3 animated dots (bounce stagger)
    - Shown while waiting for first streaming token

─────────────────────────────────────────
FIREBASE RTDB STRUCTURE:
```
{
  "users": {
    "{uid}": { "displayName": "", "email": "", "plan": "free", ... }
  },
  "chats": {
    "{uid}": {
      "{chatId}": { "id": "", "title": "", "createdAt": 0, "updatedAt": 0, "model": "gemini-2.0-flash", "messageCount": 0 }
    }
  },
  "messages": {
    "{uid}": {
      "{chatId}": {
        "{msgId}": { "id": "", "role": "user|assistant", "content": "", "timestamp": 0 }
      }
    }
  }
}
```

─────────────────────────────────────────
IMPORTANT:
- Streaming must work: tokens appear one by one in real-time
- react-markdown must render AI responses beautifully
- Syntax highlighting for code (at minimum: JS, TS, Python, Bash, JSON, HTML, CSS)
- No pink/violet-pink. Consistent premium dark UI.
- TypeScript strict, all properly typed.
- Performance: virtualize message list if >100 messages (react-window or similar)

Build all specified files completely.
```

---

---

# ═══════════════════════════════════════
# MILESTONE 4 — Modals & Settings System
# ═══════════════════════════════════════

## 🎯 PASTE THIS AS YOUR MILESTONE 4 PROMPT:

```
Continue building "VortexFlow AI". This is Milestone 4. Milestones 1-3 complete.

TASK: Build all modal components and the complete settings system.

─────────────────────────────────────────
BASE MODAL SYSTEM:

src/components/ui/Modal.tsx — Base modal wrapper:
- Framer Motion AnimatePresence (scale + opacity: 0.95→1)
- Backdrop: rgba(0,0,0,0.7) blur(4px)
- Glassmorphism card container
- Close on backdrop click + ESC key
- Sizes: sm (400px), md (560px), lg (720px), xl (900px), full
- Header: title + optional subtitle + close X button
- Body: scrollable content area
- Footer: optional action buttons slot
- Portal rendered (React createPortal to document.body)
- Focus trap inside modal

─────────────────────────────────────────
MODALS TO BUILD:

1. src/components/modals/SettingsModal.tsx (lg size):
   Tabbed layout (left vertical tabs, right content):

   TAB 1 — General:
   - Interface Language (dropdown, English only for now)
   - Default AI Model (dropdown: Gemini 2.0 Flash ⚡ / Gemini 1.5 Pro 🧠)
   - Response Language (Auto / English / other options)
   - Chat font size (Small / Medium / Large — preview text shown)
   - Show token count toggle
   - Show timestamps toggle
   - Auto-scroll to bottom toggle
   - Save changes button

   TAB 2 — AI Behaviour:
   - System Prompt textarea (custom instructions for the AI):
     - Placeholder: "You are a helpful assistant..."
     - Character count, max 2000 chars
     - "Reset to default" link
   - Creativity / Temperature slider (0.0 → 2.0):
     - Labels: "Precise" ←→ "Creative"
     - Current value shown
   - Max Response Length (Short / Medium / Long / Maximum)
   - Enable markdown rendering toggle
   - Code syntax highlighting toggle
   - Safety settings: dropdown (Default / Strict / Off — with disclaimer)

   TAB 3 — Account:
   - Profile picture (circle, shows Google avatar or gradient initials)
   - Display name (editable input + save)
   - Email (read-only, badge "Verified" or "Unverified")
   - Password section (only for email users): "Change Password" button → sub-form
   - Danger Zone (red border card):
     - "Delete all chat history" button (confirm modal)
     - "Delete Account" button (confirm modal with email re-entry)

   TAB 4 — About & Help:
   - App version
   - Links: Documentation, Report a Bug, Feature Request, Privacy Policy, Terms of Service
   - Keyboard shortcuts reference table:
     | Ctrl+K | New Chat |
     | Ctrl+B | Toggle Sidebar |
     | Ctrl+, | Open Settings |
     | Esc    | Close Modal |
     | Enter  | Send Message |
     | Shift+Enter | New Line |
   - "Built with ♥ using React + Firebase + Gemini" credit

2. src/components/modals/ProfileModal.tsx (md size):
   - Large avatar display
   - Name, email, member since date
   - Plan: "Free" badge with amber glow
   - Stats: Total Chats, Total Messages, Days Active
   - Quick links: Edit Profile → opens Settings Account tab, Sign Out
   - Activity graph (simple 7-day bar chart using CSS bars, no library needed)

3. src/components/modals/ModelPickerModal.tsx (md size):
   - Title: "Choose AI Model"
   - Model cards (selectable):
     Card 1: ⚡ Gemini 2.0 Flash
       - Speed: ████████░░ Fast
       - Quality: ███████░░░ High
       - Best for: Quick answers, coding, general use
       - Badge: "Recommended" (cyan)
     Card 2: 🧠 Gemini 1.5 Pro  
       - Speed: ██████░░░░ Moderate
       - Quality: ██████████ Excellent
       - Best for: Complex reasoning, long documents, deep analysis
       - Badge: "Most Capable"
   - Selected state: brand gradient border + glow
   - "Apply to this chat" + "Set as default" checkboxes
   - Confirm button

4. src/components/modals/DeleteConfirmModal.tsx (sm size):
   - Warning icon (red, FA triangle-exclamation, animated shake)
   - Title + description (configurable via props)
   - Optional: type-to-confirm input (for destructive actions)
   - Cancel (ghost) + Delete/Confirm (danger red) buttons
   - Loading state on confirm

5. src/components/modals/ChatRenameModal.tsx (sm size):
   - "Rename Chat" title
   - Input pre-filled with current title
   - Auto-focus + select all text
   - Enter to save, Esc to cancel
   - Save button (primary)

6. src/components/modals/ShareChatModal.tsx (md size):
   - "Share Conversation" title
   - Generate shareable link (stored in RTDB under shared/{shareId})
   - Copy link button (icon + text, success state "Copied!")
   - Share options (UI only, not functional yet): Copy Link, Twitter/X, WhatsApp
   - Privacy note: "Anyone with the link can view this conversation (read-only)"
   - Expiry: "Link valid for 7 days" (free plan)

7. src/components/modals/KeyboardShortcutsModal.tsx (md size):
   - Clean table of all keyboard shortcuts
   - Grouped: Navigation, Chat, Interface
   - Keys displayed as <kbd> styled chips

8. src/components/modals/WelcomeTourModal.tsx (lg size):
   - Shown once to new users (tracked in RTDB user node: onboardingComplete: false)
   - Multi-step tour (5 steps with progress dots):
     Step 1: Welcome + what VortexFlow AI is
     Step 2: Starting conversations + suggestion cards
     Step 3: Using models (Flash vs Pro explanation)
     Step 4: Settings + customization
     Step 5: "You're all set!" + confetti animation + "Start Chatting" button
   - Previous/Next navigation
   - "Skip tour" option

─────────────────────────────────────────
ALSO BUILD:

src/components/ui/Dropdown.tsx — Reusable dropdown menu:
- Trigger element (any child)
- Items: { label, icon, action, divider, danger, disabled }
- Right-click or click to open
- Closes on outside click + ESC
- Used in: chat item context menu, settings dropdowns

src/components/ui/Tooltip.tsx — Hover tooltip:
- top/bottom/left/right placement
- Delay: 500ms show, 0ms hide
- Dark bg, small text, FA icons support

src/hooks/useKeyboard.ts — Global keyboard shortcuts:
- Ctrl+K: new chat
- Ctrl+B: toggle sidebar
- Ctrl+,: open settings
- ESC: close any open modal (check modal stack)

─────────────────────────────────────────
STATE (add to Zustand store):
- openModal: 'settings' | 'profile' | 'model-picker' | 'delete-confirm' | 'rename' | 'share' | 'shortcuts' | 'welcome-tour' | null
- modalProps: any (config for delete confirm text, etc.)
- chatSettings per chatId: { model, temperature, systemPrompt, ... }
- globalSettings: { defaultModel, fontSize, showTokenCount, showTimestamps, autoScroll }

─────────────────────────────────────────
- All modals: Framer Motion AnimatePresence entrance/exit
- Mobile: modals go full-screen on small screens
- No pink/violet-pink. TypeScript strict.
Build all specified files completely.
```

---

---

# ═══════════════════════════════════════
# MILESTONE 5 — Advanced Chat Features
# ═══════════════════════════════════════

## 🎯 PASTE THIS AS YOUR MILESTONE 5 PROMPT:

```
Continue building "VortexFlow AI". This is Milestone 5. Milestones 1-4 complete.

TASK: Implement advanced chat features — search, message actions, regeneration, editing, image input, export, and real-time RTDB sync polish.

─────────────────────────────────────────
FEATURES TO BUILD:

1. MESSAGE EDITING (user messages):
   - Click edit icon on user message → transforms into editable textarea
   - Save: re-sends from that point, all subsequent messages replaced
   - Cancel: reverts, no change
   - RTDB: delete all messages after edited message, re-run AI

2. MESSAGE REGENERATION:
   - "Regenerate response" button on last AI message action bar
   - Deletes last AI message, re-runs with same user message
   - Loading state with TypingIndicator
   - RTDB updated accordingly

3. GLOBAL CHAT SEARCH:
   src/components/chat/SearchPanel.tsx:
   - Trigger: Ctrl+F or search icon in sidebar
   - Slide-in panel from top (full width, below navbar)
   - Search input (auto-focus)
   - Searches across: chat titles + message content (client-side, loaded chats)
   - Results grouped by chat
   - Click result: navigate to that chat + scroll to + highlight matching message
   - Debounced (300ms)
   - Shows "X results across Y chats"
   - Esc to close

4. MESSAGE COPY & EXPORT:
   - Copy button on each message: copies markdown text
   - Success state: "Copied!" with checkmark (2s)
   
   src/components/modals/ExportChatModal.tsx:
   - Formats: Markdown (.md), Plain Text (.txt), JSON (.json)
   - Preview of export content
   - Download button (browser download API)
   - "Copy to clipboard" alternative

5. LIKE / DISLIKE (feedback):
   - 👍 / 👎 buttons on AI messages
   - Stores in RTDB: feedback/{uid}/{chatId}/{msgId}: { type: "like"|"dislike", timestamp }
   - Visual toggle state (filled vs outline FA icons)
   - Like → brief green particle burst (CSS animation)
   - Dislike → optional feedback text input (tooltip popup)

6. IMAGE INPUT:
   - Paperclip button in InputBar (now enabled for image only)
   - Accepts: PNG, JPG, WEBP, GIF (max 4MB)
   - Preview thumbnail in input area with remove X
   - Sends as multimodal message to Gemini (inline_data base64)
   - Message displays image above text content
   - Drag & drop onto chat window also works
   - Error: file too large, wrong type

7. CHAT CONTEXT MENU (right-click or ··· button on chat item):
   - Rename
   - Share
   - Export
   - Duplicate (copy all messages to new chat)
   - Delete (confirm modal)
   - Pin (pinned chats shown at top of sidebar with 📌)

8. PINNED CHATS:
   - Pin/unpin from context menu
   - RTDB: chats/{uid}/{chatId}/pinned: true
   - Sidebar section "Pinned" at top, separate from date groups
   - Max 5 pinned chats (free plan)

9. MESSAGE TIMESTAMPS:
   - Per-message: show relative time on hover ("3 minutes ago")
   - Full datetime in tooltip
   - Toggle in settings (show always / hover only / never)

10. REAL-TIME SYNC INDICATOR:
    - Small status badge in Navbar: 🟢 "Synced" / 🟡 "Syncing..." / 🔴 "Offline"
    - Firebase onValue listener for connection state (.info/connected)
    - Offline: input bar shows "You're offline — messages will send when reconnected"
    - Queue messages offline, flush when reconnected

11. CHAT HEADER (above messages):
    src/components/chat/ChatHeader.tsx:
    - Current chat title (click to rename inline)
    - Model badge (click → ModelPickerModal)
    - Message count
    - Action buttons: Export, Share, ··· (more options)
    - On mobile: hamburger to open sidebar

12. AUTO-TITLE GENERATION:
    - After first AI response, generate a better title using Gemini:
      Prompt: "Generate a short 4-6 word title for this conversation. First message: '{msg}'. Respond with only the title."
    - Update RTDB + sidebar in real-time

13. TOKEN/USAGE DISPLAY:
    - After each AI response: show "~X tokens" in small muted text
    - Track per-chat total token usage in RTDB
    - In ProfileModal: show total tokens used this month

14. CODE BLOCK ENHANCEMENTS:
    - Language label (top-left of code block)
    - Copy button (top-right, shows "Copied!" 2s)
    - Line numbers toggle
    - "Run in sandbox" placeholder button (disabled, tooltip "Coming soon")
    - Collapsible for blocks >30 lines (show first 15, "Show more" button)

15. SCROLL BEHAVIOR:
    - Smart auto-scroll: if user is within 100px of bottom → auto-scroll on new token
    - If scrolled up: show "↓ New message" FAB
    - Smooth scroll (behavior: 'smooth')

─────────────────────────────────────────
PERFORMANCE:
- Message list: use react-window (VariableSizeList) if >50 messages
- Images: lazy load with IntersectionObserver
- Search: debounced + memoized with useMemo

─────────────────────────────────────────
RTDB UPDATES (additions):
- chats/{uid}/{chatId}/pinned: boolean
- messages/{uid}/{chatId}/{msgId}/edited: boolean
- messages/{uid}/{chatId}/{msgId}/imageUrl: string (base64 or null)
- feedback/{uid}/{chatId}/{msgId}: { type, timestamp, comment? }
- shared/{shareId}: { uid, chatId, createdAt, expiresAt, messages: [...] }

No pink/violet-pink. TypeScript strict. Build all specified files.
```

---

---

# ═══════════════════════════════════════
# MILESTONE 6 — UI Polish, Animations & Final Touches
# ═══════════════════════════════════════

## 🎯 PASTE THIS AS YOUR MILESTONE 6 PROMPT:

```
Continue building "VortexFlow AI". This is Milestone 6. Milestones 1-5 complete.

TASK: Final UI polish, micro-animations, performance optimizations, PWA setup, error boundaries, and production readiness.

─────────────────────────────────────────
ITEMS TO BUILD/POLISH:

1. LOADING STATES (comprehensive):
   - Full-page loader (logo animation) during auth check
   - Skeleton loaders: sidebar chat list, message area, profile
   - Inline spinners: buttons, model picker
   - Progress bar (top of page): Framer Motion, cyan gradient, appears during navigation

2. ERROR BOUNDARIES:
   src/components/ErrorBoundary.tsx:
   - Catches React render errors
   - Friendly error UI: "Something went wrong" with VortexFlow branding
   - "Reload page" + "Go to home" buttons
   - Error details (collapsed, dev mode only)
   - Wraps: ChatWindow, Sidebar, each Modal

3. EMPTY STATES (beautiful illustrations using CSS/SVG):
   - No chats yet (sidebar)
   - No search results
   - Chat load error
   - Network offline
   Each: relevant FA icon (large, gradient), headline, sub-text, optional CTA

4. NOTIFICATION SYSTEM (polish):
   - Toast stack: max 3 visible, oldest auto-dismiss
   - Notification centre dropdown (bell icon in Navbar):
     - Shows app-level notifications (welcome message, tips)
     - "No new notifications" empty state
     - Mark all read

5. ANIMATIONS (micro-interactions):
   - Send button: ripple effect on click
   - New chat created: sidebar item slides in from top
   - Chat deleted: item slides out + fades
   - Modal open: scale 0.95→1 + fade, close: reverse
   - Sidebar collapse: width transition 260→0, icon-only at 60px
   - Message appear: fade + translate-y: 8px→0
   - Streaming cursor: blinking underscore (CSS blink animation)
   - Logo glow pulse: continuous subtle animation
   - Page transitions (React Router): fade + slight upward movement

6. KEYBOARD NAVIGATION:
   - Full keyboard accessibility
   - Focus visible outlines (brand cyan, 2px)
   - Tab order: Sidebar new chat → chat list → main input → send
   - Arrow keys navigate chat list in sidebar
   - Enter opens selected chat

7. PWA SETUP:
   - vite.config.ts: vite-plugin-pwa
   - manifest.json: name, icons, theme_color (#0A0A0F), background_color, display: standalone
   - Service worker: cache shell (offline support for app UI)
   - "Install app" prompt (beforeinstallprompt event) → custom install banner

8. SEO & META:
   - index.html: OG tags, Twitter card tags, description for VortexFlow AI
   - Favicon (use logo.svg reference)
   - robots.txt (allow all except /chat/*)
   - Proper page titles per route (React Helmet or document.title)

9. PERFORMANCE:
   - React.lazy() + Suspense for: AuthPage, ChatPage, all Modals
   - Image optimization: lazy loading, proper alt texts
   - Bundle analysis: vite-bundle-visualizer (dev only)
   - Memoization: React.memo on MessageBubble, SidebarChatItem
   - useMemo: filtered/grouped chat lists
   - useCallback: all event handlers in hooks

10. ACCESSIBILITY (a11y):
    - ARIA labels on all icon buttons
    - role="dialog" + aria-modal on all modals
    - aria-live="polite" on streaming message area
    - Color contrast: all text meets WCAG AA minimum
    - Screen reader: meaningful alt texts, button labels

11. RESPONSIVE DESIGN FINAL PASS:
    - Mobile (<768px): 
      - Sidebar: hidden, opens as full overlay with backdrop
      - Input bar: larger touch targets
      - Modals: bottom sheet style (slides up)
      - Chat bubbles: full width
    - Tablet (768-1024px):
      - Sidebar: icon-only by default (60px)
      - Modals: centered, max 90vw
    - Desktop (>1024px):
      - Full sidebar
      - Wide modals

12. RATE LIMITING UI:
    - Track messages per session (Zustand)
    - Free plan: show "X / 50 messages used today" in sidebar bottom
    - Warning at 40: amber toast "You're approaching your daily limit"
    - At 50: error state on input "Daily limit reached. Limit resets at midnight."
    - (Not actually enforced via server — UI only, honesty note in FAQ)

13. CHANGELOG / WHATS NEW:
    src/components/modals/ChangelogModal.tsx:
    - Triggered from Navbar "What's New" badge (if unseen)
    - Version list with date, emoji, and description
    - Tracks last-seen version in RTDB

14. FINAL POLISH ITEMS:
    - Smooth scrollbar (thin, brand coloured, only visible on hover)
    - Focus mode: Ctrl+Shift+F hides sidebar + header for distraction-free chat
    - Print styles: @media print — clean message list, no UI chrome
    - 404 page: branded "Page not found" with nav home button
    - Consistent border-radius tokens throughout (sm: 6px, md: 10px, lg: 16px, xl: 24px)
    - All FA icons consistent size (fa-sm, fa-lg as appropriate)
    - Hover states on every interactive element (no bare clickables)
    - Selection colour: brand cyan (::selection CSS)

─────────────────────────────────────────
ENV VARIABLES DOCUMENTATION:
Create .env.example file:
```
VITE_FIREBASE_API_KEY=
VITE_FIREBASE_AUTH_DOMAIN=
VITE_FIREBASE_DATABASE_URL=
VITE_FIREBASE_PROJECT_ID=
VITE_FIREBASE_STORAGE_BUCKET=
VITE_FIREBASE_MESSAGING_SENDER_ID=
VITE_FIREBASE_APP_ID=
VITE_GEMINI_API_KEY=
```

Also create README.md with:
- Project overview
- Setup instructions (clone, install, env setup, firebase setup, run)
- Firebase RTDB rules snippet
- Deployment guide (Vercel/Netlify)
- Contributing guide placeholder

─────────────────────────────────────────
FIREBASE RTDB SECURITY RULES (include in README):
```json
{
  "rules": {
    "users": {
      "$uid": {
        ".read": "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    },
    "chats": {
      "$uid": {
        ".read": "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    },
    "messages": {
      "$uid": {
        ".read": "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    },
    "feedback": {
      "$uid": {
        ".read": "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    },
    "shared": {
      ".read": true,
      "$shareId": {
        ".write": "auth !== null"
      }
    }
  }
}
```

No pink/violet-pink. TypeScript strict. This is the final milestone — make it production-ready and perfect.
Build all specified files completely.
```

---

---

# 📋 MILESTONE SUMMARY TABLE

| Milestone | Focus | Key Deliverables |
|-----------|-------|-----------------|
| **M1** | Foundation + Landing | Project setup, design system, full landing page |
| **M2** | Authentication | Firebase Auth, Email+Google login, Auth UI |
| **M3** | Core Chat | Gemini API, RTDB, Sidebar, Chat Window, Streaming |
| **M4** | Modals & Settings | 8 modals, settings system, keyboard shortcuts |
| **M5** | Advanced Features | Edit/regen messages, search, image input, export |
| **M6** | Polish & Production | Animations, PWA, a11y, performance, README |

---

# 🖼️ LOGO PLACEMENT GUIDE

When you receive your AI-generated `logo.svg`:

1. In **AI Studio Build**, go to the **Files** panel (left sidebar)
2. Upload `logo.svg` to the `public/` folder
3. The app references it as `src="/logo.svg"` — it will appear automatically

The logo will be used in:
- Navbar (with animated glow pulse)
- Landing Page hero
- Auth page left panel
- Chat welcome screen
- Loading screen
- PWA icon (manifest)

---

# ✨ LOGO DESIGN PROMPT (For Separate AI Image/SVG Generator)

Use this prompt in an AI vector/SVG generator (e.g., Recraft AI, Adobe Firefly Vector, or ChatGPT image → trace to SVG):

```
Design a minimalist, modern logo for "VortexFlow AI".

Concept: Abstract vortex/spiral that transitions into a flowing data stream or neural connection. The spiral should feel like a whirlpool of intelligence — dynamic, elegant, and technological.

Style:
- Minimalist geometric — works at 16px favicon and 200px display
- Single shape or 2-3 interlocking elements max
- NO text in the logo mark (icon only)
- Monochrome base design that looks great filled with gradient

Colour (when exported as SVG with gradient):
- Primary gradient: #00D4FF (electric cyan) → #7B61FF (deep indigo)
- Flow: left-to-right or top-to-bottom linear gradient
- Background: transparent

Animation (the SVG will have CSS animation applied):
- The vortex spiral arms will have a slow continuous rotation (360deg, 8s, linear, infinite)
- Subtle opacity pulse on the outer ring (0.7 → 1.0, 3s ease-in-out, alternate infinite)
- On hover: rotation speeds up (3s), glow filter applied (drop-shadow cyan)

Format: SVG, clean paths, no raster elements, viewBox="0 0 64 64"

The result should feel: premium, intelligent, slightly futuristic — similar in quality to Anthropic's or OpenAI's logo aesthetic but with a vortex/flow theme.
```

---

*VortexFlow AI — Roadmap-Prompt.md — v1.0*
*Total Milestones: 6 | Stack: React + TS + Vite + Firebase + Gemini*
