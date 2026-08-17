import { PropsWithChildren } from 'react'

export function Card({ children, className = '' }: PropsWithChildren<{ className?: string }>) {
  return (
    <div className={`rounded-[28px] border border-white/10 bg-white/5 shadow-glow backdrop-blur-2xl ${className}`}>
      {children}
    </div>
  )
}
