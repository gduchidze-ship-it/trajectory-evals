import * as RT from "@radix-ui/react-tabs";
import clsx from "clsx";
import type { ComponentProps } from "react";

export const Tabs = RT.Root;

export function TabList({ className, ...p }: ComponentProps<typeof RT.List>) {
  return <RT.List className={clsx("mb-4 flex gap-1 border-b border-line", className)} {...p} />;
}

export function Tab({ className, ...p }: ComponentProps<typeof RT.Trigger>) {
  return (
    <RT.Trigger
      className={clsx(
        "-mb-px cursor-pointer border-b-2 border-transparent bg-transparent px-3 py-2 text-[14px] font-medium text-ink-2 hover:text-ink",
        "data-[state=active]:border-accent data-[state=active]:text-ink",
        className,
      )}
      {...p}
    />
  );
}

export const TabPanel = RT.Content;
